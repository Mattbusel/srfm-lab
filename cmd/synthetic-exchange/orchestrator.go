package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

// ExchangeConfig defines the full simulation configuration.
type ExchangeConfig struct {
	Symbols            []SymbolConfig       `json:"symbols"`
	AgentCounts        AgentCountConfig     `json:"agent_counts"`
	SessionDuration    Duration             `json:"session_duration"`
	TickInterval       Duration             `json:"tick_interval"`
	LatencyProfile     LatencyProfile       `json:"latency_profile"`
	CircuitBreaker     CircuitBreakerParams `json:"circuit_breaker"`
	EnableAuctions     bool                 `json:"enable_auctions"`
	AuctionDuration    Duration             `json:"auction_duration"`
	RandomSeed         int64                `json:"random_seed"`
	RecordHistory      bool                 `json:"record_history"`
	HistoryPath        string               `json:"history_path"`
	BarFrequency       Duration             `json:"bar_frequency"`
	MaxOrdersPerAgent  int                  `json:"max_orders_per_agent"`
	FeeRateMaker       float64              `json:"fee_rate_maker"`
	FeeRateTaker       float64              `json:"fee_rate_taker"`
}

// SymbolConfig describes a single tradeable instrument.
type SymbolConfig struct {
	Name      string  `json:"name"`
	TickSize  float64 `json:"tick_size"`
	LotSize   float64 `json:"lot_size"`
	InitPrice float64 `json:"init_price"`
	MaxSpread float64 `json:"max_spread"`
}

// AgentCountConfig specifies how many agents of each type to spawn.
type AgentCountConfig struct {
	MarketMakers   int `json:"market_makers"`
	NoiseTraders   int `json:"noise_traders"`
	InformedTraders int `json:"informed_traders"`
}

// LatencyProfile models network latency distributions.
type LatencyProfile struct {
	MeanUs   int `json:"mean_us"`
	StddevUs int `json:"stddev_us"`
	JitterUs int `json:"jitter_us"`
}

// CircuitBreakerParams defines when the circuit breaker fires.
type CircuitBreakerParams struct {
	Enabled        bool    `json:"enabled"`
	ThresholdPct   float64 `json:"threshold_pct"`
	WindowSeconds  int     `json:"window_seconds"`
	CooldownSeconds int    `json:"cooldown_seconds"`
	MaxTrips       int     `json:"max_trips"`
}

// Duration wraps time.Duration for JSON marshalling.
type Duration struct {
	time.Duration
}

func (d Duration) MarshalJSON() ([]byte, error) {
	return json.Marshal(d.String())
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		// try as number of seconds
		var secs float64
		if err2 := json.Unmarshal(b, &secs); err2 != nil {
			return err
		}
		d.Duration = time.Duration(secs * float64(time.Second))
		return nil
	}
	var err error
	d.Duration, err = time.ParseDuration(s)
	return err
}

// Validate checks the config for sanity.
func (c *ExchangeConfig) Validate() error {
	if len(c.Symbols) == 0 {
		return errors.New("at least one symbol required")
	}
	for i, sym := range c.Symbols {
		if sym.Name == "" {
			return fmt.Errorf("symbol %d: name required", i)
		}
		if sym.TickSize <= 0 {
			return fmt.Errorf("symbol %s: tick_size must be positive", sym.Name)
		}
		if sym.LotSize <= 0 {
			return fmt.Errorf("symbol %s: lot_size must be positive", sym.Name)
		}
		if sym.InitPrice <= 0 {
			return fmt.Errorf("symbol %s: init_price must be positive", sym.Name)
		}
	}
	total := c.AgentCounts.MarketMakers + c.AgentCounts.NoiseTraders + c.AgentCounts.InformedTraders
	if total == 0 {
		return errors.New("at least one agent required")
	}
	if c.SessionDuration.Duration <= 0 {
		c.SessionDuration.Duration = 5 * time.Minute
	}
	if c.TickInterval.Duration <= 0 {
		c.TickInterval.Duration = 100 * time.Millisecond
	}
	if c.BarFrequency.Duration <= 0 {
		c.BarFrequency.Duration = 1 * time.Second
	}
	if c.MaxOrdersPerAgent <= 0 {
		c.MaxOrdersPerAgent = 50
	}
	if c.CircuitBreaker.ThresholdPct <= 0 {
		c.CircuitBreaker.ThresholdPct = 5.0
	}
	if c.CircuitBreaker.WindowSeconds <= 0 {
		c.CircuitBreaker.WindowSeconds = 60
	}
	if c.CircuitBreaker.CooldownSeconds <= 0 {
		c.CircuitBreaker.CooldownSeconds = 30
	}
	if c.CircuitBreaker.MaxTrips <= 0 {
		c.CircuitBreaker.MaxTrips = 3
	}
	return nil
}

// ---------------------------------------------------------------------------
// Simulation state
// ---------------------------------------------------------------------------

// SimulationState is the serialisable snapshot of the simulation.
type SimulationState struct {
	SessionID       string                       `json:"session_id"`
	Running         bool                         `json:"running"`
	CurrentBar      int64                        `json:"current_bar"`
	ElapsedSec      float64                      `json:"elapsed_sec"`
	TotalTrades     int64                        `json:"total_trades"`
	TotalVolume     float64                      `json:"total_volume"`
	ActiveAgents    int                          `json:"active_agents"`
	Prices          map[string]float64           `json:"prices"`
	OrderBookDepth  map[string][2]int            `json:"orderbook_depth"` // [bids, asks]
	CircuitBreakers map[string]CircuitBreakerState `json:"circuit_breakers"`
	SessionPhase    string                       `json:"session_phase"`
}

// CircuitBreakerState tracks a per-symbol circuit breaker.
type CircuitBreakerState struct {
	Tripped      bool      `json:"tripped"`
	TripCount    int       `json:"trip_count"`
	LastTrip     time.Time `json:"last_trip,omitempty"`
	CooldownEnd  time.Time `json:"cooldown_end,omitempty"`
	RefPrice     float64   `json:"ref_price"`
	WindowPrices []float64 `json:"-"`
	WindowTimes  []time.Time `json:"-"`
}

// EventInjection describes a forced event.
type EventInjection struct {
	Type      string  `json:"type"`      // flash_crash, vol_spike, liquidity_drain, circuit_breaker_trip
	Symbol    string  `json:"symbol"`
	Magnitude float64 `json:"magnitude"` // percentage or multiplier
	DurationS float64 `json:"duration_s"`
}

// AggregateMetrics holds the metrics snapshot returned by GET /exchange/metrics.
type AggregateMetrics struct {
	UptimeSec         float64                     `json:"uptime_sec"`
	TotalBars         int64                       `json:"total_bars"`
	TotalTrades       int64                       `json:"total_trades"`
	TotalVolume       float64                     `json:"total_volume"`
	AvgBarLatencyUs   float64                     `json:"avg_bar_latency_us"`
	MaxBarLatencyUs   float64                     `json:"max_bar_latency_us"`
	OrdersPerSecond   float64                     `json:"orders_per_second"`
	TradesPerSecond   float64                     `json:"trades_per_second"`
	PerSymbol         map[string]SymbolMetrics    `json:"per_symbol"`
	AgentTypeMetrics  map[string]AgentTypeMetrics `json:"agent_type_metrics"`
	CircuitBreakerTrips int                       `json:"circuit_breaker_trips"`
}

// SymbolMetrics are per-symbol aggregate stats.
type SymbolMetrics struct {
	Trades        int64   `json:"trades"`
	Volume        float64 `json:"volume"`
	VWAP          float64 `json:"vwap"`
	High          float64 `json:"high"`
	Low           float64 `json:"low"`
	LastPrice     float64 `json:"last_price"`
	SpreadAvg     float64 `json:"spread_avg"`
	SpreadCurrent float64 `json:"spread_current"`
	Volatility    float64 `json:"volatility"`
}

// AgentTypeMetrics are aggregated stats per agent type.
type AgentTypeMetrics struct {
	Count      int     `json:"count"`
	TotalPnL   float64 `json:"total_pnl"`
	AvgPnL     float64 `json:"avg_pnl"`
	TotalOrders int64  `json:"total_orders"`
	TotalFills  int64  `json:"total_fills"`
}

// AgentPopulationSummary is returned by GET /exchange/agents.
type AgentPopulationSummary struct {
	Total          int                          `json:"total"`
	ByType         map[string]int               `json:"by_type"`
	TopPnL         []AgentBrief                 `json:"top_pnl"`
	BottomPnL      []AgentBrief                 `json:"bottom_pnl"`
	TypeMetrics    map[string]AgentTypeMetrics  `json:"type_metrics"`
}

// AgentBrief is a compact agent summary.
type AgentBrief struct {
	ID   string  `json:"id"`
	Type string  `json:"type"`
	PnL  float64 `json:"pnl"`
}

// AgentDetailInfo is returned by GET /exchange/agents/{id}.
type AgentDetailInfo struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"`
	PnL          float64            `json:"pnl"`
	RealizedPnL  float64            `json:"realized_pnl"`
	UnrealizedPnL float64           `json:"unrealized_pnl"`
	OrderCount   int64              `json:"order_count"`
	FillCount    int64              `json:"fill_count"`
	CancelCount  int64              `json:"cancel_count"`
	Positions    map[string]float64 `json:"positions"`
	AvgPrices    map[string]float64 `json:"avg_prices"`
}

// ---------------------------------------------------------------------------
// Metrics collector
// ---------------------------------------------------------------------------

type metricsCollector struct {
	mu              sync.Mutex
	startTime       time.Time
	totalBars       int64
	totalTrades     int64
	totalVolume     float64
	totalOrders     int64
	barLatencies    []float64
	perSymbolTrades map[string]int64
	perSymbolVol    map[string]float64
	perSymbolVWAP   map[string][2]float64 // [sumPriceQty, sumQty]
	perSymbolHigh   map[string]float64
	perSymbolLow    map[string]float64
	perSymbolLast   map[string]float64
	perSymbolSpread []map[string]float64
	cbTrips         int
}

func newMetricsCollector() *metricsCollector {
	return &metricsCollector{
		startTime:       time.Now(),
		perSymbolTrades: make(map[string]int64),
		perSymbolVol:    make(map[string]float64),
		perSymbolVWAP:   make(map[string][2]float64),
		perSymbolHigh:   make(map[string]float64),
		perSymbolLow:    make(map[string]float64),
		perSymbolLast:   make(map[string]float64),
	}
}

func (mc *metricsCollector) recordBar(latency time.Duration) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.totalBars++
	mc.barLatencies = append(mc.barLatencies, float64(latency.Microseconds()))
}

func (mc *metricsCollector) recordTrade(symbol string, price, qty float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.totalTrades++
	mc.totalVolume += qty
	mc.perSymbolTrades[symbol]++
	mc.perSymbolVol[symbol] += qty
	vwap := mc.perSymbolVWAP[symbol]
	vwap[0] += price * qty
	vwap[1] += qty
	mc.perSymbolVWAP[symbol] = vwap
	mc.perSymbolLast[symbol] = price
	if h, ok := mc.perSymbolHigh[symbol]; !ok || price > h {
		mc.perSymbolHigh[symbol] = price
	}
	if l, ok := mc.perSymbolLow[symbol]; !ok || price < l {
		mc.perSymbolLow[symbol] = price
	}
}

func (mc *metricsCollector) recordOrders(n int) {
	mc.mu.Lock()
	mc.totalOrders += int64(n)
	mc.mu.Unlock()
}

func (mc *metricsCollector) recordCBTrip() {
	mc.mu.Lock()
	mc.cbTrips++
	mc.mu.Unlock()
}

func (mc *metricsCollector) snapshot(books map[string]*OrderBook, agents []Agent) AggregateMetrics {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	elapsed := time.Since(mc.startTime).Seconds()
	var avgLat, maxLat float64
	for _, l := range mc.barLatencies {
		avgLat += l
		if l > maxLat {
			maxLat = l
		}
	}
	if len(mc.barLatencies) > 0 {
		avgLat /= float64(len(mc.barLatencies))
	}
	var ops, tps float64
	if elapsed > 0 {
		ops = float64(mc.totalOrders) / elapsed
		tps = float64(mc.totalTrades) / elapsed
	}

	perSym := make(map[string]SymbolMetrics)
	for sym := range mc.perSymbolTrades {
		sm := SymbolMetrics{
			Trades:    mc.perSymbolTrades[sym],
			Volume:    mc.perSymbolVol[sym],
			High:      mc.perSymbolHigh[sym],
			Low:       mc.perSymbolLow[sym],
			LastPrice: mc.perSymbolLast[sym],
		}
		if vwap := mc.perSymbolVWAP[sym]; vwap[1] > 0 {
			sm.VWAP = vwap[0] / vwap[1]
		}
		if ob, ok := books[sym]; ok {
			best := ob.BestBidAsk()
			if best[0] > 0 && best[1] > 0 {
				sm.SpreadCurrent = best[1] - best[0]
			}
		}
		perSym[sym] = sm
	}

	atm := aggregateAgentTypeMetrics(agents)

	return AggregateMetrics{
		UptimeSec:           elapsed,
		TotalBars:           mc.totalBars,
		TotalTrades:         mc.totalTrades,
		TotalVolume:         mc.totalVolume,
		AvgBarLatencyUs:     avgLat,
		MaxBarLatencyUs:     maxLat,
		OrdersPerSecond:     ops,
		TradesPerSecond:     tps,
		PerSymbol:           perSym,
		AgentTypeMetrics:    atm,
		CircuitBreakerTrips: mc.cbTrips,
	}
}

func aggregateAgentTypeMetrics(agents []Agent) map[string]AgentTypeMetrics {
	out := make(map[string]AgentTypeMetrics)
	for _, a := range agents {
		m := a.Metrics()
		at := out[a.Type()]
		at.Count++
		at.TotalPnL += m.PnL
		at.TotalOrders += m.OrderCount
		at.TotalFills += m.FillCount
		out[a.Type()] = at
	}
	for k, v := range out {
		if v.Count > 0 {
			v.AvgPnL = v.TotalPnL / float64(v.Count)
		}
		out[k] = v
	}
	return out
}

// ---------------------------------------------------------------------------
// Circuit breaker
// ---------------------------------------------------------------------------

type circuitBreakerManager struct {
	mu     sync.Mutex
	params CircuitBreakerParams
	states map[string]*CircuitBreakerState
}

func newCircuitBreakerManager(params CircuitBreakerParams) *circuitBreakerManager {
	return &circuitBreakerManager{
		params: params,
		states: make(map[string]*CircuitBreakerState),
	}
}

func (cbm *circuitBreakerManager) init(symbol string, refPrice float64) {
	cbm.mu.Lock()
	defer cbm.mu.Unlock()
	cbm.states[symbol] = &CircuitBreakerState{RefPrice: refPrice}
}

func (cbm *circuitBreakerManager) check(symbol string, price float64, now time.Time) bool {
	if !cbm.params.Enabled {
		return false
	}
	cbm.mu.Lock()
	defer cbm.mu.Unlock()
	st, ok := cbm.states[symbol]
	if !ok {
		return false
	}
	if st.Tripped {
		if now.Before(st.CooldownEnd) {
			return true
		}
		st.Tripped = false
		st.RefPrice = price
		st.WindowPrices = nil
		st.WindowTimes = nil
		return false
	}
	window := time.Duration(cbm.params.WindowSeconds) * time.Second
	cutoff := now.Add(-window)
	// trim old entries
	j := 0
	for i, t := range st.WindowTimes {
		if t.After(cutoff) {
			j = i
			break
		}
		if i == len(st.WindowTimes)-1 {
			j = i + 1
		}
	}
	st.WindowPrices = st.WindowPrices[j:]
	st.WindowTimes = st.WindowTimes[j:]
	st.WindowPrices = append(st.WindowPrices, price)
	st.WindowTimes = append(st.WindowTimes, now)

	if st.RefPrice <= 0 {
		st.RefPrice = price
		return false
	}
	movePct := math.Abs(price-st.RefPrice) / st.RefPrice * 100.0
	if movePct >= cbm.params.ThresholdPct {
		if st.TripCount >= cbm.params.MaxTrips {
			return false // max trips exhausted
		}
		st.Tripped = true
		st.TripCount++
		st.LastTrip = now
		st.CooldownEnd = now.Add(time.Duration(cbm.params.CooldownSeconds) * time.Second)
		return true
	}
	return false
}

func (cbm *circuitBreakerManager) forceTrip(symbol string, now time.Time) {
	cbm.mu.Lock()
	defer cbm.mu.Unlock()
	st, ok := cbm.states[symbol]
	if !ok {
		return
	}
	st.Tripped = true
	st.TripCount++
	st.LastTrip = now
	st.CooldownEnd = now.Add(time.Duration(cbm.params.CooldownSeconds) * time.Second)
}

func (cbm *circuitBreakerManager) stateSnapshot() map[string]CircuitBreakerState {
	cbm.mu.Lock()
	defer cbm.mu.Unlock()
	out := make(map[string]CircuitBreakerState, len(cbm.states))
	for k, v := range cbm.states {
		out[k] = *v
	}
	return out
}

// ---------------------------------------------------------------------------
// Session manager
// ---------------------------------------------------------------------------

type sessionPhase int

const (
	phasePreOpen sessionPhase = iota
	phaseOpenAuction
	phaseContinuous
	phaseCloseAuction
	phaseClosed
)

func (p sessionPhase) String() string {
	switch p {
	case phasePreOpen:
		return "pre_open"
	case phaseOpenAuction:
		return "open_auction"
	case phaseContinuous:
		return "continuous"
	case phaseCloseAuction:
		return "close_auction"
	case phaseClosed:
		return "closed"
	default:
		return "unknown"
	}
}

type sessionManager struct {
	mu              sync.Mutex
	phase           sessionPhase
	sessionStart    time.Time
	sessionDuration time.Duration
	auctionDuration time.Duration
	enableAuctions  bool
}

func newSessionManager(cfg ExchangeConfig) *sessionManager {
	return &sessionManager{
		phase:           phasePreOpen,
		sessionDuration: cfg.SessionDuration.Duration,
		auctionDuration: cfg.AuctionDuration.Duration,
		enableAuctions:  cfg.EnableAuctions,
	}
}

func (sm *sessionManager) start(now time.Time) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.sessionStart = now
	if sm.enableAuctions {
		sm.phase = phaseOpenAuction
	} else {
		sm.phase = phaseContinuous
	}
}

func (sm *sessionManager) update(now time.Time) sessionPhase {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	elapsed := now.Sub(sm.sessionStart)
	remaining := sm.sessionDuration - elapsed

	switch sm.phase {
	case phaseOpenAuction:
		if elapsed >= sm.auctionDuration {
			sm.phase = phaseContinuous
		}
	case phaseContinuous:
		if sm.enableAuctions && remaining <= sm.auctionDuration {
			sm.phase = phaseCloseAuction
		} else if remaining <= 0 {
			sm.phase = phaseClosed
		}
	case phaseCloseAuction:
		if remaining <= 0 {
			sm.phase = phaseClosed
		}
	}
	return sm.phase
}

func (sm *sessionManager) currentPhase() sessionPhase {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return sm.phase
}

func (sm *sessionManager) isTrading() bool {
	p := sm.currentPhase()
	return p == phaseContinuous || p == phaseOpenAuction || p == phaseCloseAuction
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

// Orchestrator is the main simulation engine.
type Orchestrator struct {
	mu        sync.RWMutex
	running   bool
	sessionID string
	config    ExchangeConfig
	startTime time.Time
	barCount  int64

	books      map[string]*OrderBook
	tradeLogs  map[string]*TradeLog
	agents     []Agent
	registry   *AgentRegistry
	popMgr     *PopulationManager
	metrics    *metricsCollector
	cbMgr      *circuitBreakerManager
	sessMgr    *sessionManager
	barAgg     map[string]*BarAggregator
	recorder   *HistoricalRecorder
	feed       *DataFeed

	hub    *streamHub
	cancel context.CancelFunc
	done   chan struct{}

	// event injection channel
	eventCh chan EventInjection
}

// context type for internal cancel
type context struct {
	done chan struct{}
}

func (c *context) Done() <-chan struct{} { return c.done }

type CancelFunc func()

type contextPair struct {
	ctx    *context
	cancel CancelFunc
}

func withCancel() (*context, CancelFunc) {
	ctx := &context{done: make(chan struct{})}
	return ctx, func() {
		select {
		case <-ctx.done:
		default:
			close(ctx.done)
		}
	}
}

// NewOrchestrator creates a new orchestrator connected to the SSE hub.
func NewOrchestrator(hub *streamHub) *Orchestrator {
	return &Orchestrator{
		hub:     hub,
		eventCh: make(chan EventInjection, 64),
	}
}

func genSessionID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "sim-" + hex.EncodeToString(b)
}

// SessionID returns the current session ID.
func (o *Orchestrator) SessionID() string {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.sessionID
}

// IsRunning returns true if the simulation is active.
func (o *Orchestrator) IsRunning() bool {
	o.mu.RLock()
	defer o.mu.RUnlock()
	return o.running
}

// Start initializes and starts the simulation.
func (o *Orchestrator) Start(cfg ExchangeConfig) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.running {
		return errors.New("simulation already running")
	}

	o.config = cfg
	o.sessionID = genSessionID()
	o.startTime = time.Now()
	o.barCount = 0

	// order books and trade logs
	o.books = make(map[string]*OrderBook, len(cfg.Symbols))
	o.tradeLogs = make(map[string]*TradeLog, len(cfg.Symbols))
	o.barAgg = make(map[string]*BarAggregator, len(cfg.Symbols))
	for _, sym := range cfg.Symbols {
		o.books[sym.Name] = NewOrderBook(sym.Name, sym.TickSize)
		o.tradeLogs[sym.Name] = NewTradeLog(10000)
		o.barAgg[sym.Name] = NewBarAggregator(sym.Name, cfg.BarFrequency.Duration)
	}

	// metrics
	o.metrics = newMetricsCollector()

	// circuit breakers
	o.cbMgr = newCircuitBreakerManager(cfg.CircuitBreaker)
	for _, sym := range cfg.Symbols {
		o.cbMgr.init(sym.Name, sym.InitPrice)
	}

	// session
	o.sessMgr = newSessionManager(cfg)

	// agents
	o.registry = NewAgentRegistry()
	o.popMgr = NewPopulationManager(o.registry)
	o.agents = nil
	o.initAgents(cfg)

	// feed
	o.feed = NewDataFeed(256)

	// recorder
	if cfg.RecordHistory {
		path := cfg.HistoryPath
		if path == "" {
			path = fmt.Sprintf("sim_%s.jsonl", o.sessionID)
		}
		var err error
		o.recorder, err = NewHistoricalRecorder(path)
		if err != nil {
			log.Printf("[orchestrator] warning: cannot open recorder: %v", err)
			o.recorder = nil
		}
	}

	// start loop
	ctx, cancel := withCancel()
	o.cancel = CancelFunc(cancel)
	o.done = make(chan struct{})
	o.running = true
	o.sessMgr.start(time.Now())

	go o.runLoop(ctx)
	log.Printf("[orchestrator] started session %s with %d agents, %d symbols",
		o.sessionID, len(o.agents), len(cfg.Symbols))
	return nil
}

func (o *Orchestrator) initAgents(cfg ExchangeConfig) {
	symbols := make([]string, len(cfg.Symbols))
	initPrices := make(map[string]float64)
	for i, s := range cfg.Symbols {
		symbols[i] = s.Name
		initPrices[s.Name] = s.InitPrice
	}

	for i := 0; i < cfg.AgentCounts.MarketMakers; i++ {
		a := NewGoMarketMaker(fmt.Sprintf("mm-%04d", i), symbols, initPrices, cfg)
		o.agents = append(o.agents, a)
		o.registry.Register(a)
	}
	for i := 0; i < cfg.AgentCounts.NoiseTraders; i++ {
		a := NewGoNoiseTrader(fmt.Sprintf("nt-%04d", i), symbols, initPrices, cfg)
		o.agents = append(o.agents, a)
		o.registry.Register(a)
	}
	for i := 0; i < cfg.AgentCounts.InformedTraders; i++ {
		a := NewGoInformedTrader(fmt.Sprintf("it-%04d", i), symbols, initPrices, cfg)
		o.agents = append(o.agents, a)
		o.registry.Register(a)
	}
}

// Stop halts the simulation.
func (o *Orchestrator) Stop() error {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.running {
		return errors.New("simulation not running")
	}
	o.cancel()
	<-o.done
	o.running = false
	if o.recorder != nil {
		o.recorder.Close()
	}
	log.Printf("[orchestrator] stopped session %s after %d bars", o.sessionID, o.barCount)
	return nil
}

// runLoop is the main simulation loop.
func (o *Orchestrator) runLoop(ctx *context) {
	defer close(o.done)
	ticker := time.NewTicker(o.config.TickInterval.Duration)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			phase := o.sessMgr.update(time.Now())
			if phase == phaseClosed {
				o.mu.Lock()
				o.running = false
				o.mu.Unlock()
				log.Printf("[orchestrator] session ended naturally after %d bars", o.barCount)
				return
			}
			o.runBar()
		}
	}
}

// runBar processes one simulation tick.
func (o *Orchestrator) runBar() {
	start := time.Now()
	o.mu.Lock()
	o.barCount++
	barNum := o.barCount
	o.mu.Unlock()

	// drain injected events
	for {
		select {
		case evt := <-o.eventCh:
			o.applyEvent(evt)
		default:
			goto doneEvents
		}
	}
doneEvents:

	if !o.sessMgr.isTrading() {
		return
	}

	// build market data for agents
	snapshots := make(map[string]MarketDataSnapshot)
	for sym, book := range o.books {
		snap := book.Snapshot()
		if agg, ok := o.barAgg[sym]; ok {
			snap.Bar = agg.CurrentBar()
		}
		snapshots[sym] = snap
	}

	md := MarketData{
		Bar:       barNum,
		Timestamp: time.Now(),
		Snapshots: snapshots,
	}

	// update agents and collect orders
	var allOrders []Order
	for _, a := range o.agents {
		a.OnUpdate(md)
		orders := a.GenerateOrders()
		allOrders = append(allOrders, orders...)
	}
	o.metrics.recordOrders(len(allOrders))

	// match orders per symbol
	for _, order := range allOrders {
		sym := order.Symbol
		book, ok := o.books[sym]
		if !ok {
			continue
		}
		// circuit breaker check
		if o.cbMgr != nil {
			cbState := o.cbMgr.stateSnapshot()
			if st, ok := cbState[sym]; ok && st.Tripped {
				continue // skip orders during circuit breaker
			}
		}
		fills := book.ProcessOrder(order)
		for _, fill := range fills {
			o.metrics.recordTrade(sym, fill.Price, fill.Qty)
			if tl, ok := o.tradeLogs[sym]; ok {
				tl.Add(fill)
			}
			if agg, ok := o.barAgg[sym]; ok {
				agg.AddTick(fill.Price, fill.Qty, fill.Timestamp)
			}
			// circuit breaker check on fill price
			if o.cbMgr != nil {
				tripped := o.cbMgr.check(sym, fill.Price, fill.Timestamp)
				if tripped {
					o.metrics.recordCBTrip()
					log.Printf("[circuit_breaker] tripped for %s at price %.4f", sym, fill.Price)
				}
			}
			// distribute fill to agents
			for _, a := range o.agents {
				if a.ID() == fill.BuyerID || a.ID() == fill.SellerID {
					a.OnFill(fill)
				}
			}
		}
	}

	// record history
	if o.recorder != nil {
		for sym, snap := range snapshots {
			_ = o.recorder.WriteSnapshot(barNum, sym, snap)
		}
	}

	// broadcast to SSE clients
	streamData := struct {
		Bar       int64                        `json:"bar"`
		Timestamp string                       `json:"timestamp"`
		Prices    map[string]float64           `json:"prices"`
		Spreads   map[string]float64           `json:"spreads"`
		Phase     string                       `json:"phase"`
	}{
		Bar:       barNum,
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Prices:    make(map[string]float64),
		Spreads:   make(map[string]float64),
		Phase:     o.sessMgr.currentPhase().String(),
	}
	for sym, book := range o.books {
		ba := book.BestBidAsk()
		if ba[0] > 0 && ba[1] > 0 {
			streamData.Prices[sym] = (ba[0] + ba[1]) / 2.0
			streamData.Spreads[sym] = ba[1] - ba[0]
		}
	}
	if data, err := json.Marshal(streamData); err == nil {
		o.hub.broadcast(data)
	}

	// feed
	o.feed.Publish(md)

	elapsed := time.Since(start)
	o.metrics.recordBar(elapsed)
}

// applyEvent applies an injected event.
func (o *Orchestrator) applyEvent(evt EventInjection) {
	log.Printf("[orchestrator] applying event: %s on %s mag=%.2f dur=%.1fs",
		evt.Type, evt.Symbol, evt.Magnitude, evt.DurationS)

	switch evt.Type {
	case "flash_crash":
		book, ok := o.books[evt.Symbol]
		if !ok {
			return
		}
		// Wipe the bid side partially to simulate crash
		book.DrainSide(true, evt.Magnitude/100.0)
		// inject aggressive sell orders
		crashPrice := book.LastPrice() * (1.0 - evt.Magnitude/100.0)
		crashOrder := Order{
			ID:        fmt.Sprintf("crash-%d", time.Now().UnixNano()),
			Symbol:    evt.Symbol,
			Side:      SideSell,
			Type:      OrderTypeMarket,
			Qty:       evt.Magnitude * 100,
			Price:     crashPrice,
			Timestamp: time.Now(),
			AgentID:   "system",
		}
		fills := book.ProcessOrder(crashOrder)
		for _, f := range fills {
			o.metrics.recordTrade(evt.Symbol, f.Price, f.Qty)
			if tl, ok := o.tradeLogs[evt.Symbol]; ok {
				tl.Add(f)
			}
		}

	case "vol_spike":
		// increase noise trader aggressiveness temporarily
		for _, a := range o.agents {
			if nt, ok := a.(*GoNoiseTrader); ok {
				nt.SetVolatilityMultiplier(evt.Magnitude)
			}
		}
		if evt.DurationS > 0 {
			go func() {
				time.Sleep(time.Duration(evt.DurationS * float64(time.Second)))
				for _, a := range o.agents {
					if nt, ok := a.(*GoNoiseTrader); ok {
						nt.SetVolatilityMultiplier(1.0)
					}
				}
			}()
		}

	case "liquidity_drain":
		book, ok := o.books[evt.Symbol]
		if !ok {
			return
		}
		book.DrainSide(true, evt.Magnitude/100.0)
		book.DrainSide(false, evt.Magnitude/100.0)

	case "circuit_breaker_trip":
		if o.cbMgr != nil {
			o.cbMgr.forceTrip(evt.Symbol, time.Now())
			o.metrics.recordCBTrip()
		}
	}
}

// InjectEvent queues an event for injection.
func (o *Orchestrator) InjectEvent(evt EventInjection) error {
	if !o.IsRunning() {
		return errors.New("simulation not running")
	}
	validTypes := map[string]bool{
		"flash_crash": true, "vol_spike": true,
		"liquidity_drain": true, "circuit_breaker_trip": true,
	}
	if !validTypes[evt.Type] {
		return fmt.Errorf("unknown event type: %s", evt.Type)
	}
	select {
	case o.eventCh <- evt:
		return nil
	default:
		return errors.New("event queue full")
	}
}

// GetState returns the current simulation state.
func (o *Orchestrator) GetState() (SimulationState, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	if !o.running {
		return SimulationState{}, errors.New("simulation not running")
	}

	prices := make(map[string]float64)
	depth := make(map[string][2]int)
	for sym, book := range o.books {
		ba := book.BestBidAsk()
		if ba[0] > 0 && ba[1] > 0 {
			prices[sym] = (ba[0] + ba[1]) / 2.0
		} else {
			prices[sym] = book.LastPrice()
		}
		d := book.Depth()
		depth[sym] = d
	}

	var cbStates map[string]CircuitBreakerState
	if o.cbMgr != nil {
		cbStates = o.cbMgr.stateSnapshot()
	}

	return SimulationState{
		SessionID:       o.sessionID,
		Running:         true,
		CurrentBar:      o.barCount,
		ElapsedSec:      time.Since(o.startTime).Seconds(),
		TotalTrades:     o.metrics.totalTrades,
		TotalVolume:     o.metrics.totalVolume,
		ActiveAgents:    len(o.agents),
		Prices:          prices,
		OrderBookDepth:  depth,
		CircuitBreakers: cbStates,
		SessionPhase:    o.sessMgr.currentPhase().String(),
	}, nil
}

// GetMetrics returns aggregate performance metrics.
func (o *Orchestrator) GetMetrics() AggregateMetrics {
	o.mu.RLock()
	defer o.mu.RUnlock()
	if o.metrics == nil {
		return AggregateMetrics{}
	}
	return o.metrics.snapshot(o.books, o.agents)
}

// GetOrderBook returns a serialisable order book view.
func (o *Orchestrator) GetOrderBook(symbol string) (OrderBookView, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	book, ok := o.books[symbol]
	if !ok {
		return OrderBookView{}, fmt.Errorf("unknown symbol: %s", symbol)
	}
	return book.View(20), nil
}

// GetRecentTrades returns the last n trades for a symbol.
func (o *Orchestrator) GetRecentTrades(symbol string, n int) ([]Fill, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	tl, ok := o.tradeLogs[symbol]
	if !ok {
		return nil, fmt.Errorf("unknown symbol: %s", symbol)
	}
	return tl.Last(n), nil
}

// GetAgentSummary returns a population summary.
func (o *Orchestrator) GetAgentSummary() AgentPopulationSummary {
	o.mu.RLock()
	defer o.mu.RUnlock()

	byType := make(map[string]int)
	var briefs []AgentBrief
	for _, a := range o.agents {
		byType[a.Type()]++
		m := a.Metrics()
		briefs = append(briefs, AgentBrief{ID: a.ID(), Type: a.Type(), PnL: m.PnL})
	}

	sort.Slice(briefs, func(i, j int) bool { return briefs[i].PnL > briefs[j].PnL })

	top := briefs
	if len(top) > 10 {
		top = top[:10]
	}
	bottom := make([]AgentBrief, len(briefs))
	copy(bottom, briefs)
	sort.Slice(bottom, func(i, j int) bool { return bottom[i].PnL < bottom[j].PnL })
	if len(bottom) > 10 {
		bottom = bottom[:10]
	}

	return AgentPopulationSummary{
		Total:       len(o.agents),
		ByType:      byType,
		TopPnL:      top,
		BottomPnL:   bottom,
		TypeMetrics: aggregateAgentTypeMetrics(o.agents),
	}
}

// GetAgentInfo returns detailed info for one agent.
func (o *Orchestrator) GetAgentInfo(id string) (AgentDetailInfo, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()
	a, ok := o.registry.Get(id)
	if !ok {
		return AgentDetailInfo{}, fmt.Errorf("agent not found: %s", id)
	}
	m := a.Metrics()
	return AgentDetailInfo{
		ID:            a.ID(),
		Type:          a.Type(),
		PnL:           m.PnL,
		RealizedPnL:   m.RealizedPnL,
		UnrealizedPnL: m.UnrealizedPnL,
		OrderCount:    m.OrderCount,
		FillCount:     m.FillCount,
		CancelCount:   m.CancelCount,
		Positions:     m.Positions,
		AvgPrices:     m.AvgPrices,
	}, nil
}
