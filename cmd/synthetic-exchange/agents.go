package main

import (
	"fmt"
	"math"
	mrand "math/rand"
	"sync"
	"time"
)

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

// Side represents buy or sell.
type Side int

const (
	SideBuy Side = iota
	SideSell
)

func (s Side) String() string {
	if s == SideBuy {
		return "buy"
	}
	return "sell"
}

func (s Side) MarshalJSON() ([]byte, error) {
	return []byte(`"` + s.String() + `"`), nil
}

// OrderType is limit or market.
type OrderType int

const (
	OrderTypeLimit OrderType = iota
	OrderTypeMarket
)

func (o OrderType) String() string {
	if o == OrderTypeLimit {
		return "limit"
	}
	return "market"
}

func (o OrderType) MarshalJSON() ([]byte, error) {
	return []byte(`"` + o.String() + `"`), nil
}

// Order represents a single order submitted by an agent.
type Order struct {
	ID        string    `json:"id"`
	Symbol    string    `json:"symbol"`
	Side      Side      `json:"side"`
	Type      OrderType `json:"type"`
	Price     float64   `json:"price"`
	Qty       float64   `json:"qty"`
	Timestamp time.Time `json:"timestamp"`
	AgentID   string    `json:"agent_id"`
}

// Fill represents a matched trade.
type Fill struct {
	ID        string    `json:"id"`
	Symbol    string    `json:"symbol"`
	Price     float64   `json:"price"`
	Qty       float64   `json:"qty"`
	BuyerID   string    `json:"buyer_id"`
	SellerID  string    `json:"seller_id"`
	Timestamp time.Time `json:"timestamp"`
	Aggressor Side      `json:"aggressor"`
}

// MarketData is the snapshot passed to agents each bar.
type MarketData struct {
	Bar       int64                        `json:"bar"`
	Timestamp time.Time                    `json:"timestamp"`
	Snapshots map[string]MarketDataSnapshot `json:"snapshots"`
}

// AgentMetricsData holds per-agent performance data.
type AgentMetricsData struct {
	PnL           float64            `json:"pnl"`
	RealizedPnL   float64            `json:"realized_pnl"`
	UnrealizedPnL float64            `json:"unrealized_pnl"`
	OrderCount    int64              `json:"order_count"`
	FillCount     int64              `json:"fill_count"`
	CancelCount   int64              `json:"cancel_count"`
	Positions     map[string]float64 `json:"positions"`
	AvgPrices     map[string]float64 `json:"avg_prices"`
}

// ---------------------------------------------------------------------------
// Agent interface
// ---------------------------------------------------------------------------

// Agent is the interface every trading agent must implement.
type Agent interface {
	ID() string
	Type() string
	OnUpdate(MarketData)
	GenerateOrders() []Order
	OnFill(Fill)
	Metrics() AgentMetricsData
}

// ---------------------------------------------------------------------------
// Agent registry
// ---------------------------------------------------------------------------

// AgentRegistry provides thread-safe agent lookup.
type AgentRegistry struct {
	mu     sync.RWMutex
	agents map[string]Agent
	order  []string
}

// NewAgentRegistry creates a new registry.
func NewAgentRegistry() *AgentRegistry {
	return &AgentRegistry{agents: make(map[string]Agent)}
}

// Register adds an agent.
func (r *AgentRegistry) Register(a Agent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.agents[a.ID()] = a
	r.order = append(r.order, a.ID())
}

// Unregister removes an agent.
func (r *AgentRegistry) Unregister(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.agents, id)
	for i, oid := range r.order {
		if oid == id {
			r.order = append(r.order[:i], r.order[i+1:]...)
			break
		}
	}
}

// Get returns an agent by ID.
func (r *AgentRegistry) Get(id string) (Agent, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	a, ok := r.agents[id]
	return a, ok
}

// All returns all agents in registration order.
func (r *AgentRegistry) All() []Agent {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Agent, 0, len(r.agents))
	for _, id := range r.order {
		if a, ok := r.agents[id]; ok {
			out = append(out, a)
		}
	}
	return out
}

// Count returns total registered agents.
func (r *AgentRegistry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.agents)
}

// ListByType returns agents filtered by type.
func (r *AgentRegistry) ListByType(t string) []Agent {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var out []Agent
	for _, id := range r.order {
		if a, ok := r.agents[id]; ok && a.Type() == t {
			out = append(out, a)
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Population manager
// ---------------------------------------------------------------------------

// PopulationManager handles dynamic agent lifecycle.
type PopulationManager struct {
	mu       sync.Mutex
	registry *AgentRegistry
	nextIdx  map[string]int
}

// NewPopulationManager creates a new population manager.
func NewPopulationManager(reg *AgentRegistry) *PopulationManager {
	return &PopulationManager{
		registry: reg,
		nextIdx:  make(map[string]int),
	}
}

// AddAgent dynamically adds an agent to the simulation.
func (pm *PopulationManager) AddAgent(a Agent) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.registry.Register(a)
}

// RemoveAgent removes an agent by ID.
func (pm *PopulationManager) RemoveAgent(id string) bool {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	if _, ok := pm.registry.Get(id); !ok {
		return false
	}
	pm.registry.Unregister(id)
	return true
}

// Rebalance ensures agent counts match targets, adding or removing as needed.
func (pm *PopulationManager) Rebalance(targets map[string]int, symbols []string, initPrices map[string]float64, cfg ExchangeConfig) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	current := make(map[string]int)
	for _, a := range pm.registry.All() {
		current[a.Type()]++
	}

	for typ, target := range targets {
		have := current[typ]
		if have < target {
			for i := 0; i < target-have; i++ {
				idx := pm.nextIdx[typ]
				pm.nextIdx[typ]++
				var a Agent
				switch typ {
				case "market_maker":
					a = NewGoMarketMaker(fmt.Sprintf("mm-dyn-%04d", idx), symbols, initPrices, cfg)
				case "noise_trader":
					a = NewGoNoiseTrader(fmt.Sprintf("nt-dyn-%04d", idx), symbols, initPrices, cfg)
				case "informed_trader":
					a = NewGoInformedTrader(fmt.Sprintf("it-dyn-%04d", idx), symbols, initPrices, cfg)
				}
				if a != nil {
					pm.registry.Register(a)
				}
			}
		} else if have > target {
			// remove excess from tail
			agents := pm.registry.ListByType(typ)
			for i := len(agents) - 1; i >= 0 && have > target; i-- {
				pm.registry.Unregister(agents[i].ID())
				have--
			}
		}
	}
}

// Summary returns the count of agents by type.
func (pm *PopulationManager) Summary() map[string]int {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	out := make(map[string]int)
	for _, a := range pm.registry.All() {
		out[a.Type()]++
	}
	return out
}

// ---------------------------------------------------------------------------
// Base agent with shared bookkeeping
// ---------------------------------------------------------------------------

type baseAgent struct {
	mu        sync.Mutex
	id        string
	agentType string
	symbols   []string
	rng       *mrand.Rand

	positions   map[string]float64
	avgPrices   map[string]float64
	realizedPnL float64
	orderCount  int64
	fillCount   int64
	cancelCount int64

	lastMD    MarketData
	pendingOrd []Order
}

func newBaseAgent(id, agentType string, symbols []string) baseAgent {
	return baseAgent{
		id:        id,
		agentType: agentType,
		symbols:   symbols,
		rng:       mrand.New(mrand.NewSource(time.Now().UnixNano() + int64(len(id)))),
		positions: make(map[string]float64),
		avgPrices: make(map[string]float64),
	}
}

func (b *baseAgent) ID() string   { return b.id }
func (b *baseAgent) Type() string { return b.agentType }

func (b *baseAgent) applyFill(f Fill) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.fillCount++

	var side float64
	if f.BuyerID == b.id {
		side = 1.0
	} else {
		side = -1.0
	}

	pos := b.positions[f.Symbol]
	avg := b.avgPrices[f.Symbol]
	qty := f.Qty * side

	if (pos >= 0 && side > 0) || (pos <= 0 && side < 0) {
		// adding to position
		totalQty := math.Abs(pos) + f.Qty
		if totalQty > 0 {
			avg = (avg*math.Abs(pos) + f.Price*f.Qty) / totalQty
		}
	} else {
		// reducing position
		closedQty := math.Min(math.Abs(pos), f.Qty)
		if side > 0 {
			b.realizedPnL += closedQty * (avg - f.Price) * -1
		} else {
			b.realizedPnL += closedQty * (f.Price - avg)
		}
		remaining := f.Qty - closedQty
		if remaining > 0 {
			avg = f.Price
		}
	}

	b.positions[f.Symbol] = pos + qty
	b.avgPrices[f.Symbol] = avg
}

func (b *baseAgent) computeUnrealizedPnL(prices map[string]float64) float64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	var upnl float64
	for sym, pos := range b.positions {
		if pos == 0 {
			continue
		}
		price, ok := prices[sym]
		if !ok {
			continue
		}
		avg := b.avgPrices[sym]
		upnl += pos * (price - avg)
	}
	return upnl
}

func (b *baseAgent) metricsBase() AgentMetricsData {
	b.mu.Lock()
	defer b.mu.Unlock()
	posCopy := make(map[string]float64, len(b.positions))
	avgCopy := make(map[string]float64, len(b.avgPrices))
	for k, v := range b.positions {
		posCopy[k] = v
	}
	for k, v := range b.avgPrices {
		avgCopy[k] = v
	}
	return AgentMetricsData{
		RealizedPnL: b.realizedPnL,
		OrderCount:  b.orderCount,
		FillCount:   b.fillCount,
		CancelCount: b.cancelCount,
		Positions:   posCopy,
		AvgPrices:   avgCopy,
	}
}

func (b *baseAgent) midPrice(sym string) float64 {
	if b.lastMD.Snapshots == nil {
		return 0
	}
	snap, ok := b.lastMD.Snapshots[sym]
	if !ok {
		return 0
	}
	if snap.BestBid > 0 && snap.BestAsk > 0 {
		return (snap.BestBid + snap.BestAsk) / 2.0
	}
	return snap.LastPrice
}

func (b *baseAgent) makeOrderID() string {
	b.orderCount++
	return fmt.Sprintf("%s-o%d", b.id, b.orderCount)
}

// ---------------------------------------------------------------------------
// GoMarketMaker
// ---------------------------------------------------------------------------

// GoMarketMaker posts symmetric bid/ask quotes around the mid price.
type GoMarketMaker struct {
	baseAgent
	halfSpread     map[string]float64
	quoteQty       float64
	nLevels        int
	levelSpacing   float64
	inventoryLimit float64
	skewFactor     float64
	tickSizes      map[string]float64
}

// NewGoMarketMaker creates a market maker agent.
func NewGoMarketMaker(id string, symbols []string, initPrices map[string]float64, cfg ExchangeConfig) *GoMarketMaker {
	mm := &GoMarketMaker{
		baseAgent:      newBaseAgent(id, "market_maker", symbols),
		halfSpread:     make(map[string]float64),
		quoteQty:       1.0,
		nLevels:        5,
		levelSpacing:   0.0002,
		inventoryLimit: 100.0,
		skewFactor:     0.0001,
		tickSizes:      make(map[string]float64),
	}
	for _, sym := range cfg.Symbols {
		mm.halfSpread[sym.Name] = sym.TickSize * 3
		mm.tickSizes[sym.Name] = sym.TickSize
	}
	return mm
}

func (mm *GoMarketMaker) OnUpdate(md MarketData) {
	mm.mu.Lock()
	mm.lastMD = md
	mm.mu.Unlock()
}

func (mm *GoMarketMaker) GenerateOrders() []Order {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	var orders []Order
	now := time.Now()

	for _, sym := range mm.symbols {
		mid := mm.midPrice(sym)
		if mid <= 0 {
			continue
		}

		pos := mm.positions[sym]
		skew := pos * mm.skewFactor

		hs := mm.halfSpread[sym]
		tick := mm.tickSizes[sym]
		if tick <= 0 {
			tick = 0.01
		}

		for level := 0; level < mm.nLevels; level++ {
			offset := hs + float64(level)*mm.levelSpacing*mid
			bidPrice := roundToTick(mid-offset-skew, tick)
			askPrice := roundToTick(mid+offset-skew, tick)

			if bidPrice <= 0 || askPrice <= 0 {
				continue
			}

			qty := mm.quoteQty * (1.0 + float64(level)*0.5)

			// reduce qty when near inventory limit
			if math.Abs(pos) > mm.inventoryLimit*0.8 {
				if pos > 0 {
					// reduce bid qty
					qty *= 0.3
				} else {
					qty *= 0.3
				}
			}

			orders = append(orders, Order{
				ID:        mm.makeOrderID(),
				Symbol:    sym,
				Side:      SideBuy,
				Type:      OrderTypeLimit,
				Price:     bidPrice,
				Qty:       qty,
				Timestamp: now,
				AgentID:   mm.id,
			})
			orders = append(orders, Order{
				ID:        mm.makeOrderID(),
				Symbol:    sym,
				Side:      SideSell,
				Type:      OrderTypeLimit,
				Price:     askPrice,
				Qty:       qty,
				Timestamp: now,
				AgentID:   mm.id,
			})
		}
	}
	return orders
}

func (mm *GoMarketMaker) OnFill(f Fill) {
	mm.applyFill(f)
}

func (mm *GoMarketMaker) Metrics() AgentMetricsData {
	m := mm.metricsBase()
	prices := make(map[string]float64)
	for _, sym := range mm.symbols {
		prices[sym] = mm.midPrice(sym)
	}
	m.UnrealizedPnL = mm.computeUnrealizedPnL(prices)
	m.PnL = m.RealizedPnL + m.UnrealizedPnL
	return m
}

// ---------------------------------------------------------------------------
// GoNoiseTrader
// ---------------------------------------------------------------------------

// GoNoiseTrader sends random market orders at a configurable rate.
type GoNoiseTrader struct {
	baseAgent
	tradeProb    float64
	maxQty       float64
	volMult      float64
	volMultMu    sync.Mutex
	marketBias   float64  // slight directional bias [-1, 1]
	lastPrices   map[string]float64
}

// NewGoNoiseTrader creates a noise trader agent.
func NewGoNoiseTrader(id string, symbols []string, initPrices map[string]float64, cfg ExchangeConfig) *GoNoiseTrader {
	nt := &GoNoiseTrader{
		baseAgent:  newBaseAgent(id, "noise_trader", symbols),
		tradeProb:  0.1,
		maxQty:     5.0,
		volMult:    1.0,
		lastPrices: make(map[string]float64),
	}
	for sym, p := range initPrices {
		nt.lastPrices[sym] = p
	}
	// random bias
	nt.marketBias = (nt.rng.Float64() - 0.5) * 0.2
	return nt
}

func (nt *GoNoiseTrader) OnUpdate(md MarketData) {
	nt.mu.Lock()
	nt.lastMD = md
	for sym, snap := range md.Snapshots {
		if snap.LastPrice > 0 {
			nt.lastPrices[sym] = snap.LastPrice
		}
	}
	nt.mu.Unlock()
}

func (nt *GoNoiseTrader) GenerateOrders() []Order {
	nt.mu.Lock()
	defer nt.mu.Unlock()

	var orders []Order
	now := time.Now()

	nt.volMultMu.Lock()
	vm := nt.volMult
	nt.volMultMu.Unlock()

	prob := nt.tradeProb * vm
	if prob > 1.0 {
		prob = 1.0
	}

	for _, sym := range nt.symbols {
		if nt.rng.Float64() > prob {
			continue
		}

		side := SideBuy
		r := nt.rng.Float64() + nt.marketBias
		if r < 0.5 {
			side = SideSell
		}

		qty := (nt.rng.Float64()*nt.maxQty + 0.1) * vm
		if qty < 0.01 {
			qty = 0.01
		}

		mid := nt.midPrice(sym)
		if mid <= 0 {
			mid = nt.lastPrices[sym]
		}
		if mid <= 0 {
			continue
		}

		// sometimes use limit orders slightly off mid
		oType := OrderTypeMarket
		price := 0.0
		if nt.rng.Float64() < 0.3 {
			oType = OrderTypeLimit
			offset := mid * 0.001 * (nt.rng.Float64() + 0.5)
			if side == SideBuy {
				price = mid - offset
			} else {
				price = mid + offset
			}
		}

		orders = append(orders, Order{
			ID:        nt.makeOrderID(),
			Symbol:    sym,
			Side:      side,
			Type:      oType,
			Price:     price,
			Qty:       qty,
			Timestamp: now,
			AgentID:   nt.id,
		})
	}
	return orders
}

func (nt *GoNoiseTrader) OnFill(f Fill) {
	nt.applyFill(f)
}

func (nt *GoNoiseTrader) Metrics() AgentMetricsData {
	m := nt.metricsBase()
	prices := make(map[string]float64)
	nt.mu.Lock()
	for sym := range nt.lastPrices {
		prices[sym] = nt.midPrice(sym)
		if prices[sym] <= 0 {
			prices[sym] = nt.lastPrices[sym]
		}
	}
	nt.mu.Unlock()
	m.UnrealizedPnL = nt.computeUnrealizedPnL(prices)
	m.PnL = m.RealizedPnL + m.UnrealizedPnL
	return m
}

// SetVolatilityMultiplier adjusts the noise trader's aggressiveness.
func (nt *GoNoiseTrader) SetVolatilityMultiplier(mult float64) {
	nt.volMultMu.Lock()
	nt.volMult = mult
	nt.volMultMu.Unlock()
}

// ---------------------------------------------------------------------------
// GoInformedTrader
// ---------------------------------------------------------------------------

// GoInformedTrader trades toward a privately-known "true value" that mean-reverts
// to the market price with noise.
type GoInformedTrader struct {
	baseAgent
	trueValues     map[string]float64
	signalStrength float64
	tradeThreshold float64
	maxPosition    float64
	maxQty         float64
	updateRate     float64
	meanRevertRate float64
	lastPrices     map[string]float64
}

// NewGoInformedTrader creates an informed trader.
func NewGoInformedTrader(id string, symbols []string, initPrices map[string]float64, cfg ExchangeConfig) *GoInformedTrader {
	it := &GoInformedTrader{
		baseAgent:      newBaseAgent(id, "informed_trader", symbols),
		trueValues:     make(map[string]float64),
		signalStrength: 0.002,
		tradeThreshold: 0.001,
		maxPosition:    50.0,
		maxQty:         10.0,
		updateRate:     0.05,
		meanRevertRate: 0.01,
		lastPrices:     make(map[string]float64),
	}
	for sym, p := range initPrices {
		// initial true value is initPrice + small random offset
		offset := (it.rng.Float64() - 0.5) * 2 * it.signalStrength * p
		it.trueValues[sym] = p + offset
		it.lastPrices[sym] = p
	}
	return it
}

func (it *GoInformedTrader) OnUpdate(md MarketData) {
	it.mu.Lock()
	defer it.mu.Unlock()
	it.lastMD = md

	for sym, snap := range md.Snapshots {
		mkt := snap.LastPrice
		if mkt <= 0 {
			continue
		}
		it.lastPrices[sym] = mkt

		// update true value: mean-revert to market + random innovation
		tv := it.trueValues[sym]
		tv += it.meanRevertRate * (mkt - tv)
		tv += it.signalStrength * mkt * (it.rng.Float64() - 0.5) * 2
		if tv <= 0 {
			tv = mkt * 0.99
		}
		it.trueValues[sym] = tv
	}
}

func (it *GoInformedTrader) GenerateOrders() []Order {
	it.mu.Lock()
	defer it.mu.Unlock()

	var orders []Order
	now := time.Now()

	for _, sym := range it.symbols {
		mid := it.midPrice(sym)
		if mid <= 0 {
			mid = it.lastPrices[sym]
		}
		if mid <= 0 {
			continue
		}

		tv := it.trueValues[sym]
		signal := (tv - mid) / mid

		pos := it.positions[sym]

		// only trade if signal exceeds threshold and position within limits
		if math.Abs(signal) < it.tradeThreshold {
			continue
		}

		var side Side
		if signal > 0 {
			side = SideBuy
			if pos >= it.maxPosition {
				continue
			}
		} else {
			side = SideSell
			if pos <= -it.maxPosition {
				continue
			}
		}

		// size proportional to signal strength
		qty := math.Min(it.maxQty, math.Abs(signal)/it.tradeThreshold)
		if qty < 0.1 {
			qty = 0.1
		}

		// sometimes use limit at slight edge
		oType := OrderTypeLimit
		var price float64
		edge := math.Abs(signal) * mid * 0.3
		if side == SideBuy {
			price = mid + edge*0.5
		} else {
			price = mid - edge*0.5
		}

		// occasionally just market order
		if it.rng.Float64() < 0.2 {
			oType = OrderTypeMarket
			price = 0
		}

		orders = append(orders, Order{
			ID:        it.makeOrderID(),
			Symbol:    sym,
			Side:      side,
			Type:      oType,
			Price:     price,
			Qty:       qty,
			Timestamp: now,
			AgentID:   it.id,
		})
	}
	return orders
}

func (it *GoInformedTrader) OnFill(f Fill) {
	it.applyFill(f)
}

func (it *GoInformedTrader) Metrics() AgentMetricsData {
	m := it.metricsBase()
	prices := make(map[string]float64)
	it.mu.Lock()
	for sym := range it.lastPrices {
		prices[sym] = it.midPrice(sym)
		if prices[sym] <= 0 {
			prices[sym] = it.lastPrices[sym]
		}
	}
	it.mu.Unlock()
	m.UnrealizedPnL = it.computeUnrealizedPnL(prices)
	m.PnL = m.RealizedPnL + m.UnrealizedPnL
	return m
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

func roundToTick(price, tick float64) float64 {
	if tick <= 0 {
		return price
	}
	return math.Round(price/tick) * tick
}
