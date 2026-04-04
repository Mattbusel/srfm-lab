package alerting

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// EquityCurve holds a time series of portfolio equity values.
type EquityCurve struct {
	Points []EquityPoint
}

// EquityPoint is a single time+equity sample.
type EquityPoint struct {
	Timestamp time.Time
	Equity    float64
}

// NewEquityCurve creates an EquityCurve.
func NewEquityCurve() *EquityCurve {
	return &EquityCurve{}
}

// Add appends a new data point.
func (ec *EquityCurve) Add(ts time.Time, equity float64) {
	ec.Points = append(ec.Points, EquityPoint{Timestamp: ts, Equity: equity})
}

// Returns computes the series of daily returns from equity points.
func (ec *EquityCurve) Returns() []float64 {
	if len(ec.Points) < 2 {
		return nil
	}
	// Sort by time.
	sorted := make([]EquityPoint, len(ec.Points))
	copy(sorted, ec.Points)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Timestamp.Before(sorted[j].Timestamp)
	})

	rets := make([]float64, len(sorted)-1)
	for i := 1; i < len(sorted); i++ {
		prev := sorted[i-1].Equity
		curr := sorted[i].Equity
		if prev == 0 {
			rets[i-1] = 0
		} else {
			rets[i-1] = (curr - prev) / prev
		}
	}
	return rets
}

// TotalReturn returns the total equity return from first to last point.
func (ec *EquityCurve) TotalReturn() float64 {
	if len(ec.Points) < 2 {
		return 0
	}
	first := ec.Points[0].Equity
	last := ec.Points[len(ec.Points)-1].Equity
	if first == 0 {
		return 0
	}
	return (last - first) / first
}

// MaxDrawdown returns the maximum peak-to-trough drawdown as a fraction.
func (ec *EquityCurve) MaxDrawdown() float64 {
	if len(ec.Points) == 0 {
		return 0
	}
	peak := ec.Points[0].Equity
	maxDD := 0.0
	for _, p := range ec.Points[1:] {
		if p.Equity > peak {
			peak = p.Equity
		}
		dd := (peak - p.Equity) / peak
		if dd > maxDD {
			maxDD = dd
		}
	}
	return maxDD
}

// SharpeRatio computes the annualised Sharpe ratio.
// barsPerYear is the number of data points in a year.
func (ec *EquityCurve) SharpeRatio(barsPerYear float64) float64 {
	rets := ec.Returns()
	if len(rets) == 0 {
		return 0
	}
	var sum float64
	for _, r := range rets {
		sum += r
	}
	mean := sum / float64(len(rets))
	var sq float64
	for _, r := range rets {
		d := r - mean
		sq += d * d
	}
	variance := sq / float64(len(rets))
	std := math.Sqrt(variance)
	if std == 0 {
		return 0
	}
	return mean / std * math.Sqrt(barsPerYear)
}

// SortinoRatio computes the Sortino ratio (penalises only downside volatility).
func (ec *EquityCurve) SortinoRatio(targetReturn, barsPerYear float64) float64 {
	rets := ec.Returns()
	if len(rets) == 0 {
		return 0
	}
	var sum, downSq float64
	count := 0
	for _, r := range rets {
		sum += r
		if r < targetReturn {
			d := r - targetReturn
			downSq += d * d
			count++
		}
	}
	mean := sum / float64(len(rets))
	if count == 0 {
		return 0
	}
	downStd := math.Sqrt(downSq / float64(count))
	if downStd == 0 {
		return 0
	}
	return (mean - targetReturn) / downStd * math.Sqrt(barsPerYear)
}

// CalmarRatio returns the annualised return / max drawdown.
func (ec *EquityCurve) CalmarRatio(barsPerYear float64) float64 {
	dd := ec.MaxDrawdown()
	if dd == 0 {
		return 0
	}
	rets := ec.Returns()
	if len(rets) == 0 {
		return 0
	}
	var sum float64
	for _, r := range rets {
		sum += r
	}
	annReturn := sum / float64(len(rets)) * barsPerYear
	return annReturn / dd
}

// PerformanceSummary is a human-readable summary of equity performance.
type PerformanceSummary struct {
	TotalReturn  float64
	AnnReturn    float64
	MaxDrawdown  float64
	Sharpe       float64
	Sortino      float64
	Calmar       float64
	WinRate      float64
	AvgWin       float64
	AvgLoss      float64
	ProfitFactor float64
	NumPeriods   int
}

// Summarize computes all performance metrics.
func (ec *EquityCurve) Summarize(barsPerYear float64) PerformanceSummary {
	rets := ec.Returns()
	n := float64(len(rets))
	if n == 0 {
		return PerformanceSummary{}
	}

	var sumRet, sumWin, sumLoss float64
	var wins, losses int
	for _, r := range rets {
		sumRet += r
		if r > 0 {
			wins++
			sumWin += r
		} else if r < 0 {
			losses++
			sumLoss += r
		}
	}

	avgWin := 0.0
	if wins > 0 {
		avgWin = sumWin / float64(wins)
	}
	avgLoss := 0.0
	if losses > 0 {
		avgLoss = math.Abs(sumLoss) / float64(losses)
	}
	profitFactor := 0.0
	if sumLoss != 0 {
		profitFactor = math.Abs(sumWin / sumLoss)
	}
	winRate := 0.0
	if n > 0 {
		winRate = float64(wins) / n
	}

	meanRet := sumRet / n
	annReturn := meanRet * barsPerYear

	return PerformanceSummary{
		TotalReturn:  ec.TotalReturn(),
		AnnReturn:    annReturn,
		MaxDrawdown:  ec.MaxDrawdown(),
		Sharpe:       ec.SharpeRatio(barsPerYear),
		Sortino:      ec.SortinoRatio(0, barsPerYear),
		Calmar:       ec.CalmarRatio(barsPerYear),
		WinRate:      winRate,
		AvgWin:       avgWin,
		AvgLoss:      avgLoss,
		ProfitFactor: profitFactor,
		NumPeriods:   len(rets),
	}
}

// String returns a formatted summary string.
func (ps PerformanceSummary) String() string {
	return fmt.Sprintf(
		"TotalReturn=%.2f%% AnnReturn=%.2f%% MaxDD=%.2f%% Sharpe=%.2f Sortino=%.2f Calmar=%.2f "+
			"WinRate=%.1f%% AvgWin=%.4f AvgLoss=%.4f PF=%.2f N=%d",
		ps.TotalReturn*100, ps.AnnReturn*100, ps.MaxDrawdown*100,
		ps.Sharpe, ps.Sortino, ps.Calmar,
		ps.WinRate*100, ps.AvgWin, ps.AvgLoss, ps.ProfitFactor, ps.NumPeriods,
	)
}

// PositionSizer computes position sizes based on various risk models.
type PositionSizer struct {
	AccountEquity float64
	MaxRiskPct    float64 // max % of account to risk per trade
	MaxPositions  int
}

// NewPositionSizer creates a PositionSizer.
func NewPositionSizer(equity, maxRiskPct float64, maxPositions int) *PositionSizer {
	return &PositionSizer{
		AccountEquity: equity,
		MaxRiskPct:    maxRiskPct,
		MaxPositions:  maxPositions,
	}
}

// FixedRisk returns the number of shares for a given entry/stop price pair.
func (ps *PositionSizer) FixedRisk(entryPrice, stopPrice float64) float64 {
	if entryPrice <= 0 || stopPrice <= 0 || entryPrice == stopPrice {
		return 0
	}
	riskPerShare := math.Abs(entryPrice - stopPrice)
	dollarRisk := ps.AccountEquity * ps.MaxRiskPct / 100
	shares := dollarRisk / riskPerShare
	// Cap by account equity / entry price (can't exceed 1x leverage implicitly).
	maxShares := ps.AccountEquity / entryPrice
	if shares > maxShares {
		shares = maxShares
	}
	return math.Floor(shares)
}

// VolatilityScaled returns position size scaled inversely to recent volatility.
func (ps *PositionSizer) VolatilityScaled(entryPrice, atr float64) float64 {
	if entryPrice <= 0 || atr <= 0 {
		return 0
	}
	dollarRisk := ps.AccountEquity * ps.MaxRiskPct / 100
	shares := dollarRisk / atr
	maxShares := ps.AccountEquity / entryPrice
	if shares > maxShares {
		shares = maxShares
	}
	return math.Floor(shares)
}

// KellyCriterion computes the full Kelly fraction.
func KellyCriterion(winRate, avgWin, avgLoss float64) float64 {
	if avgLoss == 0 {
		return 0
	}
	b := avgWin / avgLoss
	p := winRate
	q := 1 - p
	kelly := (b*p - q) / b
	if kelly < 0 {
		return 0
	}
	return kelly
}
