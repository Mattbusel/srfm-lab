// Package model defines the shared data structures used throughout
// the risk aggregation service. All structs are JSON-serializable
// and represent canonical risk report payloads.
package model

import "time"

// ---------------------------------------------------------------------------
// VaR structures
// ---------------------------------------------------------------------------

// VaRMethod enumerates the three supported VaR computation methods.
type VaRMethod string

const (
	VaRMethodParametric  VaRMethod = "parametric"
	VaRMethodHistorical  VaRMethod = "historical"
	VaRMethodMonteCarlo  VaRMethod = "monte_carlo"
	VaRMethodConsensus   VaRMethod = "consensus"
)

// VaRResult holds a single VaR estimate at a given confidence level.
type VaRResult struct {
	Method          VaRMethod `json:"method"`
	ConfidenceLevel float64   `json:"confidence_level"` // e.g. 0.95 or 0.99
	HorizonDays     int       `json:"horizon_days"`
	VaRAbsolute     float64   `json:"var_absolute"`  // dollar loss
	VaRPercent      float64   `json:"var_percent"`   // as fraction of NAV
	CVaR            float64   `json:"cvar_absolute"` // expected shortfall
	CVaRPercent     float64   `json:"cvar_percent"`
	ComputedAt      time.Time `json:"computed_at"`
}

// VaRConsensus bundles the three individual VaR estimates plus the
// weighted consensus and any divergence flags.
type VaRConsensus struct {
	Parametric     VaRResult `json:"parametric"`
	Historical     VaRResult `json:"historical"`
	MonteCarlo     VaRResult `json:"monte_carlo"`
	Consensus      VaRResult `json:"consensus"`
	// Weights applied (should sum to 1.0)
	WeightParam    float64   `json:"weight_parametric"`
	WeightHist     float64   `json:"weight_historical"`
	WeightMC       float64   `json:"weight_monte_carlo"`
	// Divergence is true when any pairwise relative diff exceeds threshold
	Divergent      bool      `json:"divergent"`
	DivergenceNote string    `json:"divergence_note,omitempty"`
}

// ---------------------------------------------------------------------------
// Greeks structures
// ---------------------------------------------------------------------------

// PositionGreeks holds the option Greeks for a single position.
type PositionGreeks struct {
	Instrument string  `json:"instrument"`
	Underlying string  `json:"underlying"`
	Delta      float64 `json:"delta"`       // dV/dS
	Gamma      float64 `json:"gamma"`       // d2V/dS2
	Vega       float64 `json:"vega"`        // dV/d(sigma) per 1% move
	Theta      float64 `json:"theta"`       // dV/dt per day
	Rho        float64 `json:"rho"`         // dV/dr per 1% move
	Vanna      float64 `json:"vanna"`       // d2V/dS/d(sigma)
	Volga      float64 `json:"volga"`       // d2V/d(sigma)2
	Notional   float64 `json:"notional"`    // contract notional in USD
	Quantity   float64 `json:"quantity"`    // number of contracts (signed)
}

// GreeksReport is the portfolio-level aggregated Greeks.
type GreeksReport struct {
	AsOf            time.Time        `json:"as_of"`
	Positions       []PositionGreeks `json:"positions"`
	// Portfolio-level sums (quantity-weighted)
	NetDelta        float64          `json:"net_delta"`
	NetGamma        float64          `json:"net_gamma"`
	NetVega         float64          `json:"net_vega"`
	NetTheta        float64          `json:"net_theta"`
	NetRho          float64          `json:"net_rho"`
	// Dollar Greeks -- sensitivity per $1 NAV
	DollarDelta     float64          `json:"dollar_delta"`
	DollarGamma     float64          `json:"dollar_gamma"`
	DollarVega      float64          `json:"dollar_vega"`
	DollarTheta     float64          `json:"dollar_theta"`
	NAV             float64          `json:"nav"`
}

// ---------------------------------------------------------------------------
// Stress test structures
// ---------------------------------------------------------------------------

// ScenarioShock encodes the factor shock magnitudes for one historical scenario.
type ScenarioShock struct {
	Name        string             `json:"name"`
	Description string             `json:"description"`
	// Factor -> shock magnitude (e.g. "SPX" -> -0.37 means -37%)
	Shocks      map[string]float64 `json:"shocks"`
	// Optional vol multiplier applied to all vol surfaces
	VolMultiplier float64          `json:"vol_multiplier"`
	StartDate   string             `json:"start_date"`
	EndDate     string             `json:"end_date"`
}

// StressPositionResult is the P&L impact for one position under a scenario.
type StressPositionResult struct {
	Instrument    string  `json:"instrument"`
	Underlying    string  `json:"underlying"`
	BaseValue     float64 `json:"base_value"`
	StressValue   float64 `json:"stress_value"`
	PnL           float64 `json:"pnl"`
	PnLPercent    float64 `json:"pnl_percent"`
}

// StressResult is the full outcome of running one stress scenario.
type StressResult struct {
	Scenario       ScenarioShock          `json:"scenario"`
	RunAt          time.Time              `json:"run_at"`
	Positions      []StressPositionResult `json:"positions"`
	TotalPnL       float64                `json:"total_pnl"`
	TotalPnLPct    float64                `json:"total_pnl_pct"`
	NAV            float64                `json:"nav"`
	WorstPosition  string                 `json:"worst_position"`
}

// StressSuiteResult bundles results for all scenarios in one request.
type StressSuiteResult struct {
	RunAt     time.Time      `json:"run_at"`
	Results   []StressResult `json:"results"`
	NAV       float64        `json:"nav"`
}

// ---------------------------------------------------------------------------
// Risk limits / breach structures
// ---------------------------------------------------------------------------

// LimitSeverity encodes urgency of a breach.
type LimitSeverity string

const (
	SeverityWarning  LimitSeverity = "warning"  // 80-100% of limit
	SeverityCritical LimitSeverity = "critical" // 100-120% of limit
	SeverityBreached LimitSeverity = "breached" // >120% of limit
)

// RiskLimit defines one configured limit.
type RiskLimit struct {
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Metric      string        `json:"metric"` // e.g. "var_95_1d", "net_delta", "gross_notional"
	Scope       string        `json:"scope"`  // "portfolio", "strategy:<id>", "instrument:<id>"
	HardLimit   float64       `json:"hard_limit"`
	SoftLimit   float64       `json:"soft_limit"` // warning threshold
}

// LimitBreach records one active breach at query time.
type LimitBreach struct {
	Limit       RiskLimit     `json:"limit"`
	CurrentValue float64      `json:"current_value"`
	UtilizationPct float64    `json:"utilization_pct"` // current / hard_limit * 100
	Severity    LimitSeverity `json:"severity"`
	DetectedAt  time.Time     `json:"detected_at"`
	Note        string        `json:"note,omitempty"`
}

// LimitsReport is the full snapshot of limit statuses.
type LimitsReport struct {
	AsOf      time.Time     `json:"as_of"`
	Breaches  []LimitBreach `json:"breaches"`
	TotalLimits int         `json:"total_limits"`
	BreachCount int         `json:"breach_count"`
	WarningCount int        `json:"warning_count"`
	CriticalCount int       `json:"critical_count"`
}

// ---------------------------------------------------------------------------
// Attribution structures
// ---------------------------------------------------------------------------

// BrinsonSegment holds Brinson attribution for one sector/segment bucket.
type BrinsonSegment struct {
	Segment           string  `json:"segment"`
	PortfolioWeight   float64 `json:"portfolio_weight"`
	BenchmarkWeight   float64 `json:"benchmark_weight"`
	PortfolioReturn   float64 `json:"portfolio_return"`
	BenchmarkReturn   float64 `json:"benchmark_return"`
	AllocationEffect  float64 `json:"allocation_effect"`
	SelectionEffect   float64 `json:"selection_effect"`
	InteractionEffect float64 `json:"interaction_effect"`
	TotalEffect       float64 `json:"total_effect"`
}

// FactorContribution holds factor-model attribution for one risk factor.
type FactorContribution struct {
	Factor       string  `json:"factor"`
	Exposure     float64 `json:"exposure"`   // beta / sensitivity
	FactorReturn float64 `json:"factor_return"`
	Contribution float64 `json:"contribution"` // exposure * factor_return
}

// AttributionReport aggregates Brinson and factor attribution for one day.
type AttributionReport struct {
	Date              time.Time            `json:"date"`
	TotalReturn       float64              `json:"total_return"`
	BenchmarkReturn   float64              `json:"benchmark_return"`
	ActiveReturn      float64              `json:"active_return"`
	BrinsonSegments   []BrinsonSegment     `json:"brinson_segments"`
	// Totals of Brinson effects
	TotalAllocation   float64              `json:"total_allocation"`
	TotalSelection    float64              `json:"total_selection"`
	TotalInteraction  float64              `json:"total_interaction"`
	// Factor model
	FactorContributions []FactorContribution `json:"factor_contributions"`
	FactorReturn        float64              `json:"factor_return"`
	SpecificReturn      float64              `json:"specific_return"` // residual alpha
}

// ---------------------------------------------------------------------------
// Correlation / PCA structures
// ---------------------------------------------------------------------------

// CorrelationMatrix holds the rolling Pearson matrix for a set of instruments.
type CorrelationMatrix struct {
	AsOf        time.Time   `json:"as_of"`
	WindowDays  int         `json:"window_days"`
	Instruments []string    `json:"instruments"`
	// Row-major correlation values, length = n*n
	Values      []float64   `json:"values"`
}

// PCAResult holds principal components extracted from the correlation matrix.
type PCAResult struct {
	AsOf              time.Time   `json:"as_of"`
	Instruments       []string    `json:"instruments"`
	// Eigenvalues in descending order
	Eigenvalues       []float64   `json:"eigenvalues"`
	// VarianceExplained[i] = eigenvalues[i] / sum(eigenvalues)
	VarianceExplained []float64   `json:"variance_explained"`
	CumulativeVar     []float64   `json:"cumulative_variance_explained"`
	// Loadings: rows = components, cols = instruments
	Loadings          [][]float64 `json:"loadings"`
}

// ---------------------------------------------------------------------------
// Consolidated portfolio risk
// ---------------------------------------------------------------------------

// PortfolioRiskReport is the top-level response from GET /portfolio/risk.
// It bundles VaR consensus, Greeks, and basic position summary.
type PortfolioRiskReport struct {
	AsOf       time.Time    `json:"as_of"`
	NAV        float64      `json:"nav"`
	VaR        VaRConsensus `json:"var"`
	Greeks     GreeksReport `json:"greeks"`
	Limits     LimitsReport `json:"limits"`
	Correlation CorrelationMatrix `json:"correlation"`
}
