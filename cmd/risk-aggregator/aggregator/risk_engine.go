package aggregator

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// ──────────────────────────────────────────────────────────────────────────────
// Core domain types
// ──────────────────────────────────────────────────────────────────────────────

type Position struct {
	Symbol        string  `json:"symbol"`
	Quantity      float64 `json:"quantity"`
	AvgCost       float64 `json:"avg_cost"`
	MarketValue   float64 `json:"market_value"`
	UnrealizedPnL float64 `json:"unrealized_pnl"`
	RealizedPnL   float64 `json:"realized_pnl"`
	Sector        string  `json:"sector"`
	Beta          float64 `json:"beta"`
	Delta         float64 `json:"delta"`
	Gamma         float64 `json:"gamma"`
	Vega          float64 `json:"vega"`
	Theta         float64 `json:"theta"`
	LastPrice     float64 `json:"last_price"`
	LastUpdate    int64   `json:"last_update_ns"`
}

type PortfolioSnapshot struct {
	Positions   map[string]*Position `json:"positions"`
	TotalNAV    float64              `json:"total_nav"`
	Cash        float64              `json:"cash"`
	MarginUsed  float64              `json:"margin_used"`
	BuyingPower float64              `json:"buying_power"`
	Timestamp   int64                `json:"timestamp_ns"`
}

type RiskMetrics struct {
	PortfolioVaR95  float64 `json:"portfolio_var_95"`
	PortfolioVaR99  float64 `json:"portfolio_var_99"`
	CVaR95          float64 `json:"cvar_95"`
	CVaR99          float64 `json:"cvar_99"`
	PortfolioVol    float64 `json:"portfolio_vol"`
	Beta            float64 `json:"beta"`
	Sharpe          float64 `json:"sharpe"`
	MaxDrawdown     float64 `json:"max_drawdown"`
	CurrentDrawdown float64 `json:"current_drawdown"`
	Timestamp       int64   `json:"timestamp_ns"`
}

type GreeksAggregate struct {
	TotalDelta float64 `json:"total_delta"`
	TotalGamma float64 `json:"total_gamma"`
	TotalVega  float64 `json:"total_vega"`
	TotalTheta float64 `json:"total_theta"`
}

type ComponentVaR struct {
	Symbol      string  `json:"symbol"`
	MarginalVaR float64 `json:"marginal_var"`
	ComponentPC float64 `json:"component_pct"`
	Weight      float64 `json:"weight"`
}

type StressScenario struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Shocks      map[string]float64 `json:"shocks"`
	PnLImpact   float64            `json:"pnl_impact"`
	PctImpact   float64            `json:"pct_impact"`
}

type ConcentrationMetrics struct {
	HHI                float64            `json:"hhi"`
	Top5Weight         float64            `json:"top5_weight"`
	SectorConcentration map[string]float64 `json:"sector_concentration"`
	MaxSingleName      float64            `json:"max_single_name"`
	EffectivePositions float64            `json:"effective_positions"`
}

type DrawdownState struct {
	HighWaterMark   float64   `json:"high_water_mark"`
	CurrentDrawdown float64   `json:"current_drawdown"`
	MaxDrawdown     float64   `json:"max_drawdown"`
	PeakTime        time.Time `json:"peak_time"`
	DrawdownStart   time.Time `json:"drawdown_start"`
	DurationDays    int       `json:"duration_days"`
	DeleverageLevel int       `json:"deleverage_level"` // 0=none, 1=warn, 2=reduce, 3=halt
}

type RiskLimit struct {
	Name      string  `json:"name"`
	Type      string  `json:"type"` // position, sector, var, drawdown, concentration
	Threshold float64 `json:"threshold"`
	Current   float64 `json:"current"`
	Breached  bool    `json:"breached"`
	Severity  string  `json:"severity"` // warning, hard, critical
}

type RiskViolation struct {
	Limit     RiskLimit `json:"limit"`
	Timestamp int64     `json:"timestamp_ns"`
	Message   string    `json:"message"`
}

type CorrelationEntry struct {
	SymbolA     string  `json:"symbol_a"`
	SymbolB     string  `json:"symbol_b"`
	Correlation float64 `json:"correlation"`
}

type PnLTick struct {
	Symbol       string  `json:"symbol"`
	Price        float64 `json:"price"`
	UnrealizedPL float64 `json:"unrealized_pl"`
	TotalPL      float64 `json:"total_pl"`
	Timestamp    int64   `json:"timestamp_ns"`
}

// ──────────────────────────────────────────────────────────────────────────────
// Rolling returns buffer
// ──────────────────────────────────────────────────────────────────────────────

type RollingReturns struct {
	mu       sync.RWMutex
	returns  []float64
	capacity int
	cursor   int
	full     bool
}

func NewRollingReturns(capacity int) *RollingReturns {
	return &RollingReturns{returns: make([]float64, capacity), capacity: capacity}
}

func (r *RollingReturns) Add(ret float64) {
	r.mu.Lock()
	r.returns[r.cursor] = ret
	r.cursor = (r.cursor + 1) % r.capacity
	if r.cursor == 0 {
		r.full = true
	}
	r.mu.Unlock()
}

func (r *RollingReturns) Snapshot() []float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	n := r.cursor
	if r.full {
		n = r.capacity
	}
	out := make([]float64, n)
	if r.full {
		copy(out, r.returns[r.cursor:])
		copy(out[r.capacity-r.cursor:], r.returns[:r.cursor])
	} else {
		copy(out, r.returns[:n])
	}
	return out
}

func (r *RollingReturns) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.full {
		return r.capacity
	}
	return r.cursor
}

// ──────────────────────────────────────────────────────────────────────────────
// Correlation matrix tracker
// ──────────────────────────────────────────────────────────────────────────────

type CorrelationMatrix struct {
	mu      sync.RWMutex
	returns map[string]*RollingReturns
	window  int
}

func NewCorrelationMatrix(window int) *CorrelationMatrix {
	return &CorrelationMatrix{returns: make(map[string]*RollingReturns), window: window}
}

func (cm *CorrelationMatrix) AddReturn(symbol string, ret float64) {
	cm.mu.Lock()
	if _, ok := cm.returns[symbol]; !ok {
		cm.returns[symbol] = NewRollingReturns(cm.window)
	}
	cm.returns[symbol].Add(ret)
	cm.mu.Unlock()
}

func (cm *CorrelationMatrix) Correlation(a, b string) float64 {
	cm.mu.RLock()
	ra, okA := cm.returns[a]
	rb, okB := cm.returns[b]
	cm.mu.RUnlock()
	if !okA || !okB {
		return 0
	}
	sa := ra.Snapshot()
	sb := rb.Snapshot()
	n := len(sa)
	if len(sb) < n {
		n = len(sb)
	}
	if n < 3 {
		return 0
	}
	sa = sa[len(sa)-n:]
	sb = sb[len(sb)-n:]
	return pearson(sa, sb)
}

func (cm *CorrelationMatrix) AllPairs(symbols []string) []CorrelationEntry {
	var entries []CorrelationEntry
	for i := 0; i < len(symbols); i++ {
		for j := i + 1; j < len(symbols); j++ {
			entries = append(entries, CorrelationEntry{
				SymbolA:     symbols[i],
				SymbolB:     symbols[j],
				Correlation: cm.Correlation(symbols[i], symbols[j]),
			})
		}
	}
	return entries
}

// ──────────────────────────────────────────────────────────────────────────────
// Math utilities
// ──────────────────────────────────────────────────────────────────────────────

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func variance(xs []float64) float64 {
	if len(xs) < 2 {
		return 0
	}
	m := mean(xs)
	s := 0.0
	for _, x := range xs {
		d := x - m
		s += d * d
	}
	return s / float64(len(xs)-1)
}

func stddev(xs []float64) float64 { return math.Sqrt(variance(xs)) }

func skewness(xs []float64) float64 {
	n := float64(len(xs))
	if n < 3 {
		return 0
	}
	m := mean(xs)
	sd := stddev(xs)
	if sd == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += math.Pow((x-m)/sd, 3)
	}
	return (n / ((n - 1) * (n - 2))) * s
}

func kurtosisExcess(xs []float64) float64 {
	n := float64(len(xs))
	if n < 4 {
		return 0
	}
	m := mean(xs)
	sd := stddev(xs)
	if sd == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += math.Pow((x-m)/sd, 4)
	}
	k := (n*(n+1))/((n-1)*(n-2)*(n-3))*s - 3*(n-1)*(n-1)/((n-2)*(n-3))
	return k
}

func pearson(xs, ys []float64) float64 {
	n := len(xs)
	if n < 2 || n != len(ys) {
		return 0
	}
	mx := mean(xs)
	my := mean(ys)
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

func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return sorted[0]
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

func sortedCopy(xs []float64) []float64 {
	c := make([]float64, len(xs))
	copy(c, xs)
	sort.Float64s(c)
	return c
}

func sumFloat(xs []float64) float64 {
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s
}

func absFloat(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func maxFloat(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

// ──────────────────────────────────────────────────────────────────────────────
// VaR calculators
// ──────────────────────────────────────────────────────────────────────────────

type VaRResult struct {
	VaR95          float64        `json:"var_95"`
	VaR99          float64        `json:"var_99"`
	CVaR95         float64        `json:"cvar_95"`
	CVaR99         float64        `json:"cvar_99"`
	Method         string         `json:"method"`
	ComponentVaR   []ComponentVaR `json:"component_var,omitempty"`
	ParametricAdj  *ParametricAdj `json:"parametric_adj,omitempty"`
}

type ParametricAdj struct {
	Skewness       float64 `json:"skewness"`
	KurtosisExcess float64 `json:"kurtosis_excess"`
	ZAdj95         float64 `json:"z_adj_95"`
	ZAdj99         float64 `json:"z_adj_99"`
}

// HistoricalVaR computes VaR and CVaR from rolling returns.
func HistoricalVaR(returns []float64, nav float64) VaRResult {
	if len(returns) < 10 {
		return VaRResult{Method: "historical"}
	}
	s := sortedCopy(returns)
	var95 := -percentile(s, 0.05) * nav
	var99 := -percentile(s, 0.01) * nav

	// CVaR = mean of tail beyond VaR
	cvar95 := cvarFromSorted(s, 0.05, nav)
	cvar99 := cvarFromSorted(s, 0.01, nav)

	return VaRResult{
		VaR95:  var95,
		VaR99:  var99,
		CVaR95: cvar95,
		CVaR99: cvar99,
		Method: "historical",
	}
}

func cvarFromSorted(sorted []float64, alpha float64, nav float64) float64 {
	cutoff := int(math.Floor(alpha * float64(len(sorted))))
	if cutoff < 1 {
		cutoff = 1
	}
	tail := sorted[:cutoff]
	return -mean(tail) * nav
}

// ParametricVaR with Cornish-Fisher expansion.
func ParametricVaR(returns []float64, nav float64) VaRResult {
	if len(returns) < 10 {
		return VaRResult{Method: "parametric"}
	}
	mu := mean(returns)
	sigma := stddev(returns)
	sk := skewness(returns)
	ku := kurtosisExcess(returns)

	z95 := cornishFisher(1.6449, sk, ku)
	z99 := cornishFisher(2.3263, sk, ku)

	var95 := -(mu - z95*sigma) * nav
	var99 := -(mu - z99*sigma) * nav

	// Approximate CVaR for normal: sigma * phi(z) / alpha
	phi95 := math.Exp(-z95*z95/2) / math.Sqrt(2*math.Pi)
	phi99 := math.Exp(-z99*z99/2) / math.Sqrt(2*math.Pi)
	cvar95 := (sigma*phi95/0.05 - mu) * nav
	cvar99 := (sigma*phi99/0.01 - mu) * nav

	return VaRResult{
		VaR95:  var95,
		VaR99:  var99,
		CVaR95: cvar95,
		CVaR99: cvar99,
		Method: "parametric",
		ParametricAdj: &ParametricAdj{
			Skewness:       sk,
			KurtosisExcess: ku,
			ZAdj95:         z95,
			ZAdj99:         z99,
		},
	}
}

func cornishFisher(z, sk, ku float64) float64 {
	// Cornish-Fisher expansion to fourth order
	z2 := z * z
	adj := z + (z2-1)*sk/6 + (z2*z-3*z)*ku/24 - (2*z2*z-5*z)*sk*sk/36
	return adj
}

// ComponentVaRCalc computes marginal VaR contribution per position.
func ComponentVaRCalc(positions map[string]*Position, returns map[string][]float64, totalNav float64) []ComponentVaR {
	symbols := make([]string, 0, len(positions))
	weights := make(map[string]float64)
	for sym, pos := range positions {
		symbols = append(symbols, sym)
		if totalNav > 0 {
			weights[sym] = pos.MarketValue / totalNav
		}
	}
	sort.Strings(symbols)

	// Compute portfolio variance via weight * cov * weight
	n := len(symbols)
	if n == 0 {
		return nil
	}

	// Build covariance matrix
	cov := make([][]float64, n)
	for i := 0; i < n; i++ {
		cov[i] = make([]float64, n)
		ri := returns[symbols[i]]
		for j := 0; j <= i; j++ {
			rj := returns[symbols[j]]
			c := covariance(ri, rj)
			cov[i][j] = c
			cov[j][i] = c
		}
	}

	// Portfolio variance = w' * Cov * w
	portVar := 0.0
	for i := 0; i < n; i++ {
		wi := weights[symbols[i]]
		for j := 0; j < n; j++ {
			wj := weights[symbols[j]]
			portVar += wi * wj * cov[i][j]
		}
	}
	portSigma := math.Sqrt(portVar)
	if portSigma == 0 {
		return nil
	}

	// Marginal VaR = z * (Cov * w)_i / portSigma
	z95 := 1.6449
	results := make([]ComponentVaR, n)
	for i := 0; i < n; i++ {
		covW := 0.0
		for j := 0; j < n; j++ {
			covW += cov[i][j] * weights[symbols[j]]
		}
		marginal := z95 * covW / portSigma
		component := weights[symbols[i]] * marginal * totalNav
		results[i] = ComponentVaR{
			Symbol:      symbols[i],
			MarginalVaR: marginal * totalNav,
			ComponentPC: component / (z95 * portSigma * totalNav) * 100,
			Weight:      weights[symbols[i]],
		}
	}
	return results
}

func covariance(xs, ys []float64) float64 {
	n := len(xs)
	if len(ys) < n {
		n = len(ys)
	}
	if n < 2 {
		return 0
	}
	mx := mean(xs[:n])
	my := mean(ys[:n])
	s := 0.0
	for i := 0; i < n; i++ {
		s += (xs[i] - mx) * (ys[i] - my)
	}
	return s / float64(n-1)
}

// ──────────────────────────────────────────────────────────────────────────────
// Greeks aggregation
// ──────────────────────────────────────────────────────────────────────────────

func AggregateGreeks(positions map[string]*Position) GreeksAggregate {
	var g GreeksAggregate
	for _, p := range positions {
		g.TotalDelta += p.Delta * p.Quantity
		g.TotalGamma += p.Gamma * p.Quantity
		g.TotalVega += p.Vega * p.Quantity
		g.TotalTheta += p.Theta * p.Quantity
	}
	return g
}

// ──────────────────────────────────────────────────────────────────────────────
// Stress testing (8 scenarios)
// ──────────────────────────────────────────────────────────────────────────────

var DefaultStressScenarios = []StressScenario{
	{Name: "rate_shock_up", Description: "Interest rates +200bp", Shocks: map[string]float64{"equity": -0.08, "bond": -0.12, "reit": -0.15, "tech": -0.10}},
	{Name: "rate_shock_down", Description: "Interest rates -150bp", Shocks: map[string]float64{"equity": 0.04, "bond": 0.08, "reit": 0.06, "tech": 0.05}},
	{Name: "equity_crash", Description: "Equity market -25%", Shocks: map[string]float64{"equity": -0.25, "tech": -0.35, "financials": -0.30, "healthcare": -0.15, "utilities": -0.08}},
	{Name: "vol_spike", Description: "VIX to 60", Shocks: map[string]float64{"equity": -0.12, "options_vega": 0.40, "options_gamma": 0.20, "bond": 0.02}},
	{Name: "credit_crisis", Description: "Credit spreads +500bp", Shocks: map[string]float64{"equity": -0.18, "financials": -0.35, "bond_hy": -0.25, "bond_ig": -0.08}},
	{Name: "correlation_spike", Description: "All correlations to 0.95", Shocks: map[string]float64{"equity": -0.15, "diversification_benefit": -0.50}},
	{Name: "liquidity_crisis", Description: "Bid-ask widens 10x", Shocks: map[string]float64{"equity": -0.10, "small_cap": -0.25, "cost_multiplier": 10.0}},
	{Name: "stagflation", Description: "Inflation +5%, GDP -3%", Shocks: map[string]float64{"equity": -0.15, "bond": -0.10, "commodity": 0.20, "tips": 0.05, "gold": 0.15}},
}

func RunStressScenarios(positions map[string]*Position, nav float64, scenarios []StressScenario) []StressScenario {
	results := make([]StressScenario, len(scenarios))
	for i, sc := range scenarios {
		results[i] = sc
		pnl := 0.0
		for _, pos := range positions {
			sector := strings.ToLower(pos.Sector)
			shock := sc.Shocks["equity"] // default
			if s, ok := sc.Shocks[sector]; ok {
				shock = s
			}
			positionPnL := pos.MarketValue * shock
			// Apply vega shock if present
			if vegaShock, ok := sc.Shocks["options_vega"]; ok {
				positionPnL += pos.Vega * pos.Quantity * vegaShock
			}
			if gammaShock, ok := sc.Shocks["options_gamma"]; ok {
				positionPnL += pos.Gamma * pos.Quantity * gammaShock * pos.LastPrice
			}
			pnl += positionPnL
		}
		results[i].PnLImpact = pnl
		if nav > 0 {
			results[i].PctImpact = pnl / nav
		}
	}
	return results
}

// ──────────────────────────────────────────────────────────────────────────────
// Concentration metrics
// ──────────────────────────────────────────────────────────────────────────────

func ComputeConcentration(positions map[string]*Position, nav float64) ConcentrationMetrics {
	if nav <= 0 || len(positions) == 0 {
		return ConcentrationMetrics{}
	}
	weights := make([]float64, 0, len(positions))
	sectorWeights := make(map[string]float64)
	for _, p := range positions {
		w := absFloat(p.MarketValue) / nav
		weights = append(weights, w)
		sectorWeights[p.Sector] += w
	}
	sort.Float64s(weights)

	// HHI
	hhi := 0.0
	for _, w := range weights {
		hhi += w * w
	}

	// Top 5
	top5 := 0.0
	n := len(weights)
	for i := maxInt(0, n-5); i < n; i++ {
		top5 += weights[i]
	}

	// Effective positions = 1/HHI
	eff := 0.0
	if hhi > 0 {
		eff = 1.0 / hhi
	}

	maxSingle := 0.0
	if n > 0 {
		maxSingle = weights[n-1]
	}

	return ConcentrationMetrics{
		HHI:                 hhi,
		Top5Weight:          top5,
		SectorConcentration: sectorWeights,
		MaxSingleName:       maxSingle,
		EffectivePositions:  eff,
	}
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ──────────────────────────────────────────────────────────────────────────────
// Drawdown monitor
// ──────────────────────────────────────────────────────────────────────────────

type DrawdownMonitor struct {
	mu    sync.Mutex
	state DrawdownState
	// Deleverage triggers: level 1 = -5%, 2 = -10%, 3 = -20%
	levels [3]float64
}

func NewDrawdownMonitor() *DrawdownMonitor {
	return &DrawdownMonitor{
		levels: [3]float64{-0.05, -0.10, -0.20},
	}
}

func (dm *DrawdownMonitor) Update(nav float64) DrawdownState {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	now := time.Now()
	if nav > dm.state.HighWaterMark {
		dm.state.HighWaterMark = nav
		dm.state.PeakTime = now
		dm.state.DrawdownStart = time.Time{}
	}

	if dm.state.HighWaterMark > 0 {
		dd := (nav - dm.state.HighWaterMark) / dm.state.HighWaterMark
		dm.state.CurrentDrawdown = dd
		if dd < dm.state.MaxDrawdown {
			dm.state.MaxDrawdown = dd
		}
		if dd < 0 && dm.state.DrawdownStart.IsZero() {
			dm.state.DrawdownStart = now
		}
		if !dm.state.DrawdownStart.IsZero() {
			dm.state.DurationDays = int(now.Sub(dm.state.DrawdownStart).Hours() / 24)
		}

		dm.state.DeleverageLevel = 0
		for i, lvl := range dm.levels {
			if dd <= lvl {
				dm.state.DeleverageLevel = i + 1
			}
		}
	}
	return dm.state
}

func (dm *DrawdownMonitor) State() DrawdownState {
	dm.mu.Lock()
	defer dm.mu.Unlock()
	return dm.state
}

// ──────────────────────────────────────────────────────────────────────────────
// Risk limits
// ──────────────────────────────────────────────────────────────────────────────

type RiskLimits struct {
	MaxPositionPct  float64 `json:"max_position_pct"`
	MaxSectorPct    float64 `json:"max_sector_pct"`
	MaxVaR95Pct     float64 `json:"max_var95_pct"`
	MaxDrawdownPct  float64 `json:"max_drawdown_pct"`
	MaxHHI          float64 `json:"max_hhi"`
	MaxLeverage     float64 `json:"max_leverage"`
	MaxBeta         float64 `json:"max_beta"`
	MaxCorrelation  float64 `json:"max_correlation"`
}

func DefaultRiskLimits() RiskLimits {
	return RiskLimits{
		MaxPositionPct: 0.10,
		MaxSectorPct:   0.30,
		MaxVaR95Pct:    0.02,
		MaxDrawdownPct: -0.15,
		MaxHHI:         0.15,
		MaxLeverage:    2.0,
		MaxBeta:        1.5,
		MaxCorrelation: 0.85,
	}
}

func CheckLimits(snap *PortfolioSnapshot, metrics *RiskMetrics, conc ConcentrationMetrics, dd DrawdownState, limits RiskLimits) []RiskViolation {
	var violations []RiskViolation
	now := time.Now().UnixNano()

	// Position concentration
	for sym, pos := range snap.Positions {
		if snap.TotalNAV > 0 {
			pct := absFloat(pos.MarketValue) / snap.TotalNAV
			if pct > limits.MaxPositionPct {
				violations = append(violations, RiskViolation{
					Limit: RiskLimit{
						Name: fmt.Sprintf("position_%s", sym), Type: "position",
						Threshold: limits.MaxPositionPct, Current: pct,
						Breached: true, Severity: severityForBreach(pct, limits.MaxPositionPct),
					},
					Timestamp: now,
					Message:   fmt.Sprintf("Position %s at %.1f%% exceeds limit %.1f%%", sym, pct*100, limits.MaxPositionPct*100),
				})
			}
		}
	}

	// Sector concentration
	sectors := make(map[string]float64)
	for _, pos := range snap.Positions {
		sectors[pos.Sector] += absFloat(pos.MarketValue)
	}
	for sector, val := range sectors {
		if snap.TotalNAV > 0 {
			pct := val / snap.TotalNAV
			if pct > limits.MaxSectorPct {
				violations = append(violations, RiskViolation{
					Limit: RiskLimit{
						Name: fmt.Sprintf("sector_%s", sector), Type: "sector",
						Threshold: limits.MaxSectorPct, Current: pct,
						Breached: true, Severity: severityForBreach(pct, limits.MaxSectorPct),
					},
					Timestamp: now,
					Message:   fmt.Sprintf("Sector %s at %.1f%% exceeds limit %.1f%%", sector, pct*100, limits.MaxSectorPct*100),
				})
			}
		}
	}

	// VaR limit
	if snap.TotalNAV > 0 {
		varPct := metrics.PortfolioVaR95 / snap.TotalNAV
		if varPct > limits.MaxVaR95Pct {
			violations = append(violations, RiskViolation{
				Limit: RiskLimit{
					Name: "portfolio_var95", Type: "var",
					Threshold: limits.MaxVaR95Pct, Current: varPct,
					Breached: true, Severity: severityForBreach(varPct, limits.MaxVaR95Pct),
				},
				Timestamp: now,
				Message:   fmt.Sprintf("Portfolio VaR95 at %.2f%% exceeds limit %.2f%%", varPct*100, limits.MaxVaR95Pct*100),
			})
		}
	}

	// Drawdown limit
	if dd.CurrentDrawdown < limits.MaxDrawdownPct {
		violations = append(violations, RiskViolation{
			Limit: RiskLimit{
				Name: "max_drawdown", Type: "drawdown",
				Threshold: limits.MaxDrawdownPct, Current: dd.CurrentDrawdown,
				Breached: true, Severity: "critical",
			},
			Timestamp: now,
			Message:   fmt.Sprintf("Drawdown at %.2f%% exceeds limit %.2f%%", dd.CurrentDrawdown*100, limits.MaxDrawdownPct*100),
		})
	}

	// HHI limit
	if conc.HHI > limits.MaxHHI {
		violations = append(violations, RiskViolation{
			Limit: RiskLimit{
				Name: "concentration_hhi", Type: "concentration",
				Threshold: limits.MaxHHI, Current: conc.HHI,
				Breached: true, Severity: "warning",
			},
			Timestamp: now,
			Message:   fmt.Sprintf("HHI at %.4f exceeds limit %.4f", conc.HHI, limits.MaxHHI),
		})
	}

	// Leverage
	if snap.TotalNAV > 0 && snap.MarginUsed/snap.TotalNAV > limits.MaxLeverage {
		lev := snap.MarginUsed / snap.TotalNAV
		violations = append(violations, RiskViolation{
			Limit: RiskLimit{
				Name: "leverage", Type: "concentration",
				Threshold: limits.MaxLeverage, Current: lev,
				Breached: true, Severity: "hard",
			},
			Timestamp: now,
			Message: fmt.Sprintf("Leverage %.2fx exceeds limit %.2fx", lev, limits.MaxLeverage),
		})
	}

	// Beta
	if absFloat(metrics.Beta) > limits.MaxBeta {
		violations = append(violations, RiskViolation{
			Limit: RiskLimit{
				Name: "portfolio_beta", Type: "var",
				Threshold: limits.MaxBeta, Current: metrics.Beta,
				Breached: true, Severity: "warning",
			},
			Timestamp: now,
			Message: fmt.Sprintf("Portfolio beta %.2f exceeds limit %.2f", metrics.Beta, limits.MaxBeta),
		})
	}

	return violations
}

func severityForBreach(current, threshold float64) string {
	ratio := current / threshold
	if ratio > 1.5 {
		return "critical"
	}
	if ratio > 1.2 {
		return "hard"
	}
	return "warning"
}

// ──────────────────────────────────────────────────────────────────────────────
// Real-time P&L streaming
// ──────────────────────────────────────────────────────────────────────────────

type PnLStream struct {
	mu          sync.RWMutex
	subscribers map[int64]chan PnLTick
	nextID      int64
}

func NewPnLStream() *PnLStream {
	return &PnLStream{subscribers: make(map[int64]chan PnLTick)}
}

func (ps *PnLStream) Subscribe() (int64, <-chan PnLTick) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	id := ps.nextID
	ps.nextID++
	ch := make(chan PnLTick, 256)
	ps.subscribers[id] = ch
	return id, ch
}

func (ps *PnLStream) Unsubscribe(id int64) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if ch, ok := ps.subscribers[id]; ok {
		close(ch)
		delete(ps.subscribers, id)
	}
}

func (ps *PnLStream) Publish(tick PnLTick) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	for _, ch := range ps.subscribers {
		select {
		case ch <- tick:
		default: // drop if slow
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Risk engine (orchestrator)
// ──────────────────────────────────────────────────────────────────────────────

type RiskEngine struct {
	mu              sync.RWMutex
	snapshot        PortfolioSnapshot
	metrics         RiskMetrics
	limits          RiskLimits
	violations      []RiskViolation
	rollingReturns  *RollingReturns
	assetReturns    map[string]*RollingReturns
	corrMatrix      *CorrelationMatrix
	drawdownMon     *DrawdownMonitor
	pnlStream       *PnLStream
	stressResults   []StressScenario
	concentration   ConcentrationMetrics
	greeks          GreeksAggregate
	componentVaR    []ComponentVaR
	prevNAV         float64
	lastCalcTime    time.Time
}

func NewRiskEngine() *RiskEngine {
	return &RiskEngine{
		snapshot:       PortfolioSnapshot{Positions: make(map[string]*Position)},
		limits:         DefaultRiskLimits(),
		rollingReturns: NewRollingReturns(252),
		assetReturns:   make(map[string]*RollingReturns),
		corrMatrix:     NewCorrelationMatrix(21),
		drawdownMon:    NewDrawdownMonitor(),
		pnlStream:      NewPnLStream(),
	}
}

// UpdatePosition handles a tick-level update for a single position.
func (re *RiskEngine) UpdatePosition(pos Position) {
	re.mu.Lock()
	existing, exists := re.snapshot.Positions[pos.Symbol]
	if !exists {
		p := pos
		re.snapshot.Positions[pos.Symbol] = &p
	} else {
		prevMV := existing.MarketValue
		*existing = pos
		// Compute return for correlation tracking
		if prevMV != 0 {
			ret := (pos.MarketValue - prevMV) / absFloat(prevMV)
			if _, ok := re.assetReturns[pos.Symbol]; !ok {
				re.assetReturns[pos.Symbol] = NewRollingReturns(252)
			}
			re.assetReturns[pos.Symbol].Add(ret)
			re.corrMatrix.AddReturn(pos.Symbol, ret)
		}
	}
	re.mu.Unlock()

	// Publish P&L tick
	re.pnlStream.Publish(PnLTick{
		Symbol:       pos.Symbol,
		Price:        pos.LastPrice,
		UnrealizedPL: pos.UnrealizedPnL,
		TotalPL:      pos.UnrealizedPnL + pos.RealizedPnL,
		Timestamp:    time.Now().UnixNano(),
	})
}

// SetCash sets cash balance.
func (re *RiskEngine) SetCash(cash float64) {
	re.mu.Lock()
	re.snapshot.Cash = cash
	re.mu.Unlock()
}

// SetMargin sets margin used.
func (re *RiskEngine) SetMargin(margin float64) {
	re.mu.Lock()
	re.snapshot.MarginUsed = margin
	re.mu.Unlock()
}

// Recalculate recomputes all risk metrics. Should be called periodically.
func (re *RiskEngine) Recalculate() {
	re.mu.Lock()
	defer re.mu.Unlock()

	now := time.Now()

	// Compute NAV
	totalMV := 0.0
	for _, p := range re.snapshot.Positions {
		totalMV += p.MarketValue
	}
	re.snapshot.TotalNAV = totalMV + re.snapshot.Cash
	re.snapshot.BuyingPower = re.snapshot.TotalNAV - re.snapshot.MarginUsed
	re.snapshot.Timestamp = now.UnixNano()

	// Portfolio return
	if re.prevNAV > 0 {
		portRet := (re.snapshot.TotalNAV - re.prevNAV) / re.prevNAV
		re.rollingReturns.Add(portRet)
	}
	re.prevNAV = re.snapshot.TotalNAV

	// Historical VaR
	rets := re.rollingReturns.Snapshot()
	hvar := HistoricalVaR(rets, re.snapshot.TotalNAV)
	pvar := ParametricVaR(rets, re.snapshot.TotalNAV)

	// Use parametric as primary (with Cornish-Fisher)
	re.metrics.PortfolioVaR95 = pvar.VaR95
	re.metrics.PortfolioVaR99 = pvar.VaR99
	re.metrics.CVaR95 = hvar.CVaR95 // historical CVaR more robust
	re.metrics.CVaR99 = hvar.CVaR99

	// Vol, Sharpe
	if len(rets) > 1 {
		re.metrics.PortfolioVol = stddev(rets) * math.Sqrt(252)
		mu := mean(rets) * 252
		if re.metrics.PortfolioVol > 0 {
			re.metrics.Sharpe = mu / re.metrics.PortfolioVol
		}
	}

	// Beta (weighted average of position betas)
	if re.snapshot.TotalNAV > 0 {
		wb := 0.0
		for _, p := range re.snapshot.Positions {
			wb += (p.MarketValue / re.snapshot.TotalNAV) * p.Beta
		}
		re.metrics.Beta = wb
	}

	// Drawdown
	ddState := re.drawdownMon.Update(re.snapshot.TotalNAV)
	re.metrics.MaxDrawdown = ddState.MaxDrawdown
	re.metrics.CurrentDrawdown = ddState.CurrentDrawdown
	re.metrics.Timestamp = now.UnixNano()

	// Greeks
	re.greeks = AggregateGreeks(re.snapshot.Positions)

	// Concentration
	re.concentration = ComputeConcentration(re.snapshot.Positions, re.snapshot.TotalNAV)

	// Component VaR
	assetRetSnaps := make(map[string][]float64)
	for sym, rr := range re.assetReturns {
		assetRetSnaps[sym] = rr.Snapshot()
	}
	re.componentVaR = ComponentVaRCalc(re.snapshot.Positions, assetRetSnaps, re.snapshot.TotalNAV)

	// Stress scenarios
	re.stressResults = RunStressScenarios(re.snapshot.Positions, re.snapshot.TotalNAV, DefaultStressScenarios)

	// Check limits
	re.violations = CheckLimits(&re.snapshot, &re.metrics, re.concentration, ddState, re.limits)

	re.lastCalcTime = now
}

// Getters (thread-safe)

func (re *RiskEngine) Snapshot() PortfolioSnapshot {
	re.mu.RLock()
	defer re.mu.RUnlock()
	s := re.snapshot
	s.Positions = make(map[string]*Position, len(re.snapshot.Positions))
	for k, v := range re.snapshot.Positions {
		p := *v
		s.Positions[k] = &p
	}
	return s
}

func (re *RiskEngine) Metrics() RiskMetrics {
	re.mu.RLock()
	defer re.mu.RUnlock()
	return re.metrics
}

func (re *RiskEngine) Violations() []RiskViolation {
	re.mu.RLock()
	defer re.mu.RUnlock()
	out := make([]RiskViolation, len(re.violations))
	copy(out, re.violations)
	return out
}

func (re *RiskEngine) StressResults() []StressScenario {
	re.mu.RLock()
	defer re.mu.RUnlock()
	out := make([]StressScenario, len(re.stressResults))
	copy(out, re.stressResults)
	return out
}

func (re *RiskEngine) Greeks() GreeksAggregate {
	re.mu.RLock()
	defer re.mu.RUnlock()
	return re.greeks
}

func (re *RiskEngine) Concentration() ConcentrationMetrics {
	re.mu.RLock()
	defer re.mu.RUnlock()
	return re.concentration
}

func (re *RiskEngine) Drawdown() DrawdownState {
	return re.drawdownMon.State()
}

func (re *RiskEngine) VaRDetails() (VaRResult, VaRResult, []ComponentVaR) {
	re.mu.RLock()
	defer re.mu.RUnlock()
	rets := re.rollingReturns.Snapshot()
	hvar := HistoricalVaR(rets, re.snapshot.TotalNAV)
	pvar := ParametricVaR(rets, re.snapshot.TotalNAV)
	comp := make([]ComponentVaR, len(re.componentVaR))
	copy(comp, re.componentVaR)
	return hvar, pvar, comp
}

func (re *RiskEngine) CorrelationPairs() []CorrelationEntry {
	re.mu.RLock()
	symbols := make([]string, 0, len(re.snapshot.Positions))
	for s := range re.snapshot.Positions {
		symbols = append(symbols, s)
	}
	re.mu.RUnlock()
	sort.Strings(symbols)
	return re.corrMatrix.AllPairs(symbols)
}

func (re *RiskEngine) SetLimits(l RiskLimits) {
	re.mu.Lock()
	re.limits = l
	re.mu.Unlock()
}

// RunCustomScenario runs a user-defined stress scenario.
func (re *RiskEngine) RunCustomScenario(sc StressScenario) StressScenario {
	re.mu.RLock()
	positions := make(map[string]*Position, len(re.snapshot.Positions))
	for k, v := range re.snapshot.Positions {
		p := *v
		positions[k] = &p
	}
	nav := re.snapshot.TotalNAV
	re.mu.RUnlock()
	results := RunStressScenarios(positions, nav, []StressScenario{sc})
	if len(results) > 0 {
		return results[0]
	}
	return sc
}

// ──────────────────────────────────────────────────────────────────────────────
// HTTP handlers
// ──────────────────────────────────────────────────────────────────────────────

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func readJSON(r *http.Request, v interface{}) error {
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(v)
}

type PortfolioResponse struct {
	Snapshot      PortfolioSnapshot    `json:"snapshot"`
	Metrics       RiskMetrics          `json:"metrics"`
	Greeks        GreeksAggregate      `json:"greeks"`
	Concentration ConcentrationMetrics `json:"concentration"`
	Drawdown      DrawdownState        `json:"drawdown"`
	Correlations  []CorrelationEntry   `json:"correlations"`
}

type VaRResponse struct {
	Historical   VaRResult      `json:"historical"`
	Parametric   VaRResult      `json:"parametric"`
	ComponentVaR []ComponentVaR `json:"component_var"`
}

type StressResponse struct {
	Scenarios []StressScenario `json:"scenarios"`
	TotalNAV  float64          `json:"total_nav"`
	Timestamp int64            `json:"timestamp_ns"`
}

type LimitsResponse struct {
	Limits     RiskLimits      `json:"limits"`
	Violations []RiskViolation `json:"violations"`
	Clean      bool            `json:"clean"`
}

type CustomScenarioRequest struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	Shocks      map[string]float64 `json:"shocks"`
}

// RegisterHandlers mounts risk HTTP handlers on the given mux.
func (re *RiskEngine) RegisterHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/risk/portfolio", re.handlePortfolio)
	mux.HandleFunc("/risk/var", re.handleVaR)
	mux.HandleFunc("/risk/stress", re.handleStress)
	mux.HandleFunc("/risk/limits", re.handleLimits)
	mux.HandleFunc("/risk/scenario", re.handleScenario)
	mux.HandleFunc("/risk/greeks", re.handleGreeks)
	mux.HandleFunc("/risk/concentration", re.handleConcentration)
	mux.HandleFunc("/risk/drawdown", re.handleDrawdown)
	mux.HandleFunc("/risk/correlations", re.handleCorrelations)
	mux.HandleFunc("/risk/pnl/stream", re.handlePnLStream)
}

func (re *RiskEngine) handlePortfolio(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	resp := PortfolioResponse{
		Snapshot:      re.Snapshot(),
		Metrics:       re.Metrics(),
		Greeks:        re.Greeks(),
		Concentration: re.Concentration(),
		Drawdown:      re.Drawdown(),
		Correlations:  re.CorrelationPairs(),
	}
	writeJSON(w, http.StatusOK, resp)
}

func (re *RiskEngine) handleVaR(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	hvar, pvar, comp := re.VaRDetails()
	writeJSON(w, http.StatusOK, VaRResponse{Historical: hvar, Parametric: pvar, ComponentVaR: comp})
}

func (re *RiskEngine) handleStress(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, StressResponse{
		Scenarios: re.StressResults(),
		TotalNAV:  re.Snapshot().TotalNAV,
		Timestamp: time.Now().UnixNano(),
	})
}

func (re *RiskEngine) handleLimits(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	v := re.Violations()
	re.mu.RLock()
	l := re.limits
	re.mu.RUnlock()
	writeJSON(w, http.StatusOK, LimitsResponse{Limits: l, Violations: v, Clean: len(v) == 0})
}

func (re *RiskEngine) handleScenario(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req CustomScenarioRequest
	if err := readJSON(r, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	sc := StressScenario{Name: req.Name, Description: req.Description, Shocks: req.Shocks}
	result := re.RunCustomScenario(sc)
	writeJSON(w, http.StatusOK, result)
}

func (re *RiskEngine) handleGreeks(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, re.Greeks())
}

func (re *RiskEngine) handleConcentration(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, re.Concentration())
}

func (re *RiskEngine) handleDrawdown(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, re.Drawdown())
}

func (re *RiskEngine) handleCorrelations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, re.CorrelationPairs())
}

// handlePnLStream provides Server-Sent Events for real-time P&L.
func (re *RiskEngine) handlePnLStream(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id, ch := re.pnlStream.Subscribe()
	defer re.pnlStream.Unsubscribe(id)

	ctx := r.Context()
	for {
		select {
		case <-ctx.Done():
			return
		case tick, ok := <-ch:
			if !ok {
				return
			}
			data, _ := json.Marshal(tick)
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Background risk calculation loop
// ──────────────────────────────────────────────────────────────────────────────

func (re *RiskEngine) RunLoop(interval time.Duration, stop <-chan struct{}) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			re.Recalculate()
		}
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Portfolio-level utilities
// ──────────────────────────────────────────────────────────────────────────────

// GrossExposure returns sum of absolute market values.
func (re *RiskEngine) GrossExposure() float64 {
	re.mu.RLock()
	defer re.mu.RUnlock()
	gross := 0.0
	for _, p := range re.snapshot.Positions {
		gross += absFloat(p.MarketValue)
	}
	return gross
}

// NetExposure returns sum of signed market values.
func (re *RiskEngine) NetExposure() float64 {
	re.mu.RLock()
	defer re.mu.RUnlock()
	net := 0.0
	for _, p := range re.snapshot.Positions {
		net += p.MarketValue
	}
	return net
}

// LongShortRatio returns long_mv / short_mv.
func (re *RiskEngine) LongShortRatio() float64 {
	re.mu.RLock()
	defer re.mu.RUnlock()
	var longMV, shortMV float64
	for _, p := range re.snapshot.Positions {
		if p.MarketValue >= 0 {
			longMV += p.MarketValue
		} else {
			shortMV += absFloat(p.MarketValue)
		}
	}
	if shortMV == 0 {
		return math.Inf(1)
	}
	return longMV / shortMV
}

// SectorBreakdown returns market value by sector.
func (re *RiskEngine) SectorBreakdown() map[string]float64 {
	re.mu.RLock()
	defer re.mu.RUnlock()
	m := make(map[string]float64)
	for _, p := range re.snapshot.Positions {
		m[p.Sector] += p.MarketValue
	}
	return m
}

// TotalUnrealizedPnL sums unrealized P&L across all positions.
func (re *RiskEngine) TotalUnrealizedPnL() float64 {
	re.mu.RLock()
	defer re.mu.RUnlock()
	total := 0.0
	for _, p := range re.snapshot.Positions {
		total += p.UnrealizedPnL
	}
	return total
}

// TotalRealizedPnL sums realized P&L across all positions.
func (re *RiskEngine) TotalRealizedPnL() float64 {
	re.mu.RLock()
	defer re.mu.RUnlock()
	total := 0.0
	for _, p := range re.snapshot.Positions {
		total += p.RealizedPnL
	}
	return total
}

// RemovePosition removes a closed position.
func (re *RiskEngine) RemovePosition(symbol string) {
	re.mu.Lock()
	delete(re.snapshot.Positions, symbol)
	re.mu.Unlock()
}

// PositionCount returns the number of active positions.
func (re *RiskEngine) PositionCount() int {
	re.mu.RLock()
	defer re.mu.RUnlock()
	return len(re.snapshot.Positions)
}

// SymbolList returns sorted list of held symbols.
func (re *RiskEngine) SymbolList() []string {
	re.mu.RLock()
	defer re.mu.RUnlock()
	syms := make([]string, 0, len(re.snapshot.Positions))
	for s := range re.snapshot.Positions {
		syms = append(syms, s)
	}
	sort.Strings(syms)
	return syms
}

// ──────────────────────────────────────────────────────────────────────────────
// Risk attribution
// ──────────────────────────────────────────────────────────────────────────────

type RiskAttribution struct {
	Symbol     string  `json:"symbol"`
	Weight     float64 `json:"weight"`
	VarContrib float64 `json:"var_contribution"`
	VolContrib float64 `json:"vol_contribution"`
	BetaContrib float64 `json:"beta_contribution"`
	PnLContrib  float64 `json:"pnl_contribution"`
}

func (re *RiskEngine) Attribution() []RiskAttribution {
	re.mu.RLock()
	defer re.mu.RUnlock()

	nav := re.snapshot.TotalNAV
	if nav <= 0 {
		return nil
	}

	attrs := make([]RiskAttribution, 0, len(re.snapshot.Positions))
	totalUPnL := 0.0
	for _, p := range re.snapshot.Positions {
		totalUPnL += p.UnrealizedPnL
	}

	for sym, p := range re.snapshot.Positions {
		w := p.MarketValue / nav
		var varC float64
		for _, cv := range re.componentVaR {
			if cv.Symbol == sym {
				varC = cv.ComponentPC / 100
				break
			}
		}
		pnlC := 0.0
		if totalUPnL != 0 {
			pnlC = p.UnrealizedPnL / absFloat(totalUPnL)
		}
		attrs = append(attrs, RiskAttribution{
			Symbol:      sym,
			Weight:      w,
			VarContrib:  varC,
			VolContrib:  w * w, // simplified
			BetaContrib: w * p.Beta,
			PnLContrib:  pnlC,
		})
	}

	sort.Slice(attrs, func(i, j int) bool {
		return absFloat(attrs[i].VarContrib) > absFloat(attrs[j].VarContrib)
	})
	return attrs
}

// ──────────────────────────────────────────────────────────────────────────────
// Scenario P&L distribution
// ──────────────────────────────────────────────────────────────────────────────

type ScenarioPnLDist struct {
	Mean   float64   `json:"mean"`
	Median float64   `json:"median"`
	P5     float64   `json:"p5"`
	P25    float64   `json:"p25"`
	P75    float64   `json:"p75"`
	P95    float64   `json:"p95"`
	Min    float64   `json:"min"`
	Max    float64   `json:"max"`
	Values []float64 `json:"values,omitempty"`
}

// MonteCarloVaR runs simple Monte Carlo using historical returns bootstrap.
func (re *RiskEngine) MonteCarloVaR(nSims int, horizon int) ScenarioPnLDist {
	rets := re.rollingReturns.Snapshot()
	if len(rets) < 5 {
		return ScenarioPnLDist{}
	}
	re.mu.RLock()
	nav := re.snapshot.TotalNAV
	re.mu.RUnlock()

	pnls := make([]float64, nSims)
	for i := 0; i < nSims; i++ {
		cumRet := 1.0
		for d := 0; d < horizon; d++ {
			idx := lcgRand(int64(i*horizon+d)) % int64(len(rets))
			if idx < 0 {
				idx = -idx
			}
			cumRet *= (1 + rets[idx])
		}
		pnls[i] = (cumRet - 1) * nav
	}
	sorted := sortedCopy(pnls)
	return ScenarioPnLDist{
		Mean:   mean(pnls),
		Median: percentile(sorted, 0.50),
		P5:     percentile(sorted, 0.05),
		P25:    percentile(sorted, 0.25),
		P75:    percentile(sorted, 0.75),
		P95:    percentile(sorted, 0.95),
		Min:    sorted[0],
		Max:    sorted[len(sorted)-1],
	}
}

// Simple LCG pseudo-random for deterministic simulation (no math/rand import needed).
func lcgRand(seed int64) int64 {
	return (seed*6364136223846793005 + 1442695040888963407) >> 16
}

// ──────────────────────────────────────────────────────────────────────────────
// Margin calculator
// ──────────────────────────────────────────────────────────────────────────────

type MarginRequirement struct {
	Symbol      string  `json:"symbol"`
	InitialReq  float64 `json:"initial_req"`
	MaintReq    float64 `json:"maint_req"`
	CurrentVal  float64 `json:"current_val"`
}

func (re *RiskEngine) MarginRequirements() (total float64, perPosition []MarginRequirement) {
	re.mu.RLock()
	defer re.mu.RUnlock()
	for sym, p := range re.snapshot.Positions {
		mv := absFloat(p.MarketValue)
		initRate := 0.50 // Reg T default
		maintRate := 0.25
		// Concentrated positions get higher margin
		if re.snapshot.TotalNAV > 0 && mv/re.snapshot.TotalNAV > 0.10 {
			initRate = 0.70
			maintRate = 0.40
		}
		init := mv * initRate
		maint := mv * maintRate
		total += maint
		perPosition = append(perPosition, MarginRequirement{
			Symbol: sym, InitialReq: init, MaintReq: maint, CurrentVal: mv,
		})
	}
	sort.Slice(perPosition, func(i, j int) bool {
		return perPosition[i].MaintReq > perPosition[j].MaintReq
	})
	return
}

// ──────────────────────────────────────────────────────────────────────────────
// Risk summary for dashboards
// ──────────────────────────────────────────────────────────────────────────────

type RiskSummary struct {
	NAV             float64              `json:"nav"`
	Cash            float64              `json:"cash"`
	GrossExposure   float64              `json:"gross_exposure"`
	NetExposure     float64              `json:"net_exposure"`
	Leverage        float64              `json:"leverage"`
	VaR95           float64              `json:"var_95"`
	VaR99           float64              `json:"var_99"`
	Sharpe          float64              `json:"sharpe"`
	MaxDrawdown     float64              `json:"max_drawdown"`
	CurrentDrawdown float64              `json:"current_drawdown"`
	Beta            float64              `json:"beta"`
	PositionCount   int                  `json:"position_count"`
	ViolationCount  int                  `json:"violation_count"`
	HHI             float64              `json:"hhi"`
	Top5Weight      float64              `json:"top5_weight"`
	Greeks          GreeksAggregate      `json:"greeks"`
	Timestamp       int64                `json:"timestamp_ns"`
}

func (re *RiskEngine) Summary() RiskSummary {
	re.mu.RLock()
	defer re.mu.RUnlock()
	gross := 0.0
	net := 0.0
	for _, p := range re.snapshot.Positions {
		gross += absFloat(p.MarketValue)
		net += p.MarketValue
	}
	lev := 0.0
	if re.snapshot.TotalNAV > 0 {
		lev = gross / re.snapshot.TotalNAV
	}
	return RiskSummary{
		NAV:             re.snapshot.TotalNAV,
		Cash:            re.snapshot.Cash,
		GrossExposure:   gross,
		NetExposure:     net,
		Leverage:        lev,
		VaR95:           re.metrics.PortfolioVaR95,
		VaR99:           re.metrics.PortfolioVaR99,
		Sharpe:          re.metrics.Sharpe,
		MaxDrawdown:     re.metrics.MaxDrawdown,
		CurrentDrawdown: re.metrics.CurrentDrawdown,
		Beta:            re.metrics.Beta,
		PositionCount:   len(re.snapshot.Positions),
		ViolationCount:  len(re.violations),
		HHI:             re.concentration.HHI,
		Top5Weight:      re.concentration.Top5Weight,
		Greeks:          re.greeks,
		Timestamp:       time.Now().UnixNano(),
	}
}
