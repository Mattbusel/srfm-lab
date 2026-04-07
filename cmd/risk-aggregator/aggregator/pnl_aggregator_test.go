package aggregator

import (
	"testing"
)

// epsilon is the tolerance for floating-point comparisons in tests.
const epsilon = 1e-6

func approxEqual(a, b float64) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff < epsilon
}

// -- PnLAggregator tests --

func TestPnLAggregator_UnrealizedPnL_SingleBuy(t *testing.T) {
	a := NewPnLAggregator()
	a.Update("AAPL", 10, 150.0)
	a.MarkPrice("AAPL", 160.0)
	upnl := a.UnrealizedPnL()
	want := 10 * (160 - 150.0)
	if !approxEqual(upnl["AAPL"], want) {
		t.Errorf("unrealized AAPL: got %.4f want %.4f", upnl["AAPL"], want)
	}
}

func TestPnLAggregator_RealizedPnL_FullSell(t *testing.T) {
	a := NewPnLAggregator()
	a.Update("AAPL", 10, 150.0)
	a.Update("AAPL", -10, 170.0)
	rpnl := a.RealizedPnL()
	want := 10 * (170 - 150.0)
	if !approxEqual(rpnl["AAPL"], want) {
		t.Errorf("realized AAPL: got %.4f want %.4f", rpnl["AAPL"], want)
	}
}

func TestPnLAggregator_RealizedPnL_PartialSell(t *testing.T) {
	a := NewPnLAggregator()
	a.Update("AAPL", 10, 100.0)
	a.Update("AAPL", -4, 120.0) // sell 4 @ 120: realized = 4*20 = 80
	rpnl := a.RealizedPnL()
	want := 4 * 20.0
	if !approxEqual(rpnl["AAPL"], want) {
		t.Errorf("partial sell realized: got %.4f want %.4f", rpnl["AAPL"], want)
	}
	// remaining 6 lots still open
	a.MarkPrice("AAPL", 110.0)
	upnl := a.UnrealizedPnL()
	wantU := 6 * 10.0
	if !approxEqual(upnl["AAPL"], wantU) {
		t.Errorf("partial sell unrealized: got %.4f want %.4f", upnl["AAPL"], wantU)
	}
}

func TestPnLAggregator_FIFO_MultipleLots(t *testing.T) {
	// Buy 5 @ 100, then 5 @ 120, then sell 7 @ 130
	// FIFO: sell 5 @ 100 (P&L = 5*30 = 150) + sell 2 @ 120 (P&L = 2*10 = 20) = 170
	a := NewPnLAggregator()
	a.Update("X", 5, 100.0)
	a.Update("X", 5, 120.0)
	a.Update("X", -7, 130.0)
	rpnl := a.RealizedPnL()
	want := 5*30.0 + 2*10.0
	if !approxEqual(rpnl["X"], want) {
		t.Errorf("FIFO multi-lot: got %.4f want %.4f", rpnl["X"], want)
	}
}

func TestPnLAggregator_Attribution_BySector(t *testing.T) {
	a := NewPnLAggregator()
	a.SetSectorMap(map[string]string{
		"AAPL": "technology",
		"JPM":  "financials",
	})
	a.Update("AAPL", 10, 100.0)
	a.Update("JPM", 5, 200.0)
	a.MarkPrice("AAPL", 110.0) // unrealized +100
	a.MarkPrice("JPM", 190.0)  // unrealized -50

	attr := a.Attribution()
	if !approxEqual(attr.BySymbol["AAPL"], 100.0) {
		t.Errorf("attribution AAPL: got %.4f want 100", attr.BySymbol["AAPL"])
	}
	if !approxEqual(attr.BySymbol["JPM"], -50.0) {
		t.Errorf("attribution JPM: got %.4f want -50", attr.BySymbol["JPM"])
	}
	if !approxEqual(attr.BySector["technology"], 100.0) {
		t.Errorf("sector tech: got %.4f want 100", attr.BySector["technology"])
	}
	if !approxEqual(attr.BySector["financials"], -50.0) {
		t.Errorf("sector fin: got %.4f want -50", attr.BySector["financials"])
	}
	if !approxEqual(attr.Total, 50.0) {
		t.Errorf("total: got %.4f want 50", attr.Total)
	}
}

func TestPnLAggregator_ZeroCostBasisOnNoPrice(t *testing.T) {
	a := NewPnLAggregator()
	a.Update("BTC", 1, 60000.0)
	// No MarkPrice call -- unrealized should be 0 because latestPrice is set
	// by Update itself (it stores price as latestPrice).
	// Actually Update stores price into latestPrice, so unrealized = 0 at cost.
	a.MarkPrice("BTC", 60000.0) // mark == cost
	upnl := a.UnrealizedPnL()
	if !approxEqual(upnl["BTC"], 0.0) {
		t.Errorf("unrealized at cost: got %.4f want 0", upnl["BTC"])
	}
}

func TestPnLAggregator_MultipleSymbols(t *testing.T) {
	a := NewPnLAggregator()
	a.Update("A", 1, 100)
	a.Update("B", 2, 50)
	a.MarkPrice("A", 110) // +10
	a.MarkPrice("B", 60)  // +20
	total := a.YTDPnL()
	if !approxEqual(total, 30.0) {
		t.Errorf("multi-symbol YTD: got %.4f want 30", total)
	}
}

// -- ExposureCalculator tests --

func TestExposureCalculator_GrossExposure(t *testing.T) {
	ec := &ExposureCalculator{}
	cases := []struct {
		name      string
		positions map[string]float64
		want      float64
	}{
		{"all longs", map[string]float64{"A": 100, "B": 200}, 300},
		{"mixed", map[string]float64{"A": 100, "B": -50}, 150},
		{"all shorts", map[string]float64{"A": -100, "B": -200}, 300},
		{"empty", map[string]float64{}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ec.GrossExposure(tc.positions)
			if !approxEqual(got, tc.want) {
				t.Errorf("got %.4f want %.4f", got, tc.want)
			}
		})
	}
}

func TestExposureCalculator_NetExposure(t *testing.T) {
	ec := &ExposureCalculator{}
	got := ec.NetExposure(map[string]float64{"A": 100, "B": -50})
	if !approxEqual(got, 50.0) {
		t.Errorf("net: got %.4f want 50", got)
	}
}

func TestExposureCalculator_SectorExposure(t *testing.T) {
	ec := &ExposureCalculator{}
	positions := map[string]float64{"AAPL": 300, "MSFT": 200, "JPM": 100}
	sectorMap := map[string]string{"AAPL": "tech", "MSFT": "tech", "JPM": "financials"}
	got := ec.SectorExposure(positions, sectorMap)
	if !approxEqual(got["tech"], 500) {
		t.Errorf("tech: got %.4f want 500", got["tech"])
	}
	if !approxEqual(got["financials"], 100) {
		t.Errorf("fin: got %.4f want 100", got["financials"])
	}
}

func TestExposureCalculator_BetaAdjusted(t *testing.T) {
	ec := &ExposureCalculator{}
	positions := map[string]float64{"AAPL": 100, "SPY": 200}
	betas := map[string]float64{"AAPL": 1.2, "SPY": 1.0}
	got := ec.BetaAdjustedExposure(positions, betas)
	want := 100*1.2 + 200*1.0
	if !approxEqual(got, want) {
		t.Errorf("beta adjusted: got %.4f want %.4f", got, want)
	}
}

func TestExposureCalculator_BetaAdjusted_DefaultBeta(t *testing.T) {
	ec := &ExposureCalculator{}
	// Symbol with no beta entry should default to 1.0
	positions := map[string]float64{"UNKN": 500}
	got := ec.BetaAdjustedExposure(positions, map[string]float64{})
	if !approxEqual(got, 500) {
		t.Errorf("default beta: got %.4f want 500", got)
	}
}

func TestExposureCalculator_DollarDelta(t *testing.T) {
	ec := &ExposureCalculator{}
	opts := []OptionPosition{
		{Symbol: "AAPL", Delta: 0.5, Notional: 10000},
		{Symbol: "AAPL", Delta: -0.3, Notional: 5000},
	}
	got := ec.DollarDelta(opts)
	want := 0.5*10000 + (-0.3)*5000
	if !approxEqual(got, want) {
		t.Errorf("dollar delta: got %.4f want %.4f", got, want)
	}
}

func TestExposureCalculator_Leverage(t *testing.T) {
	ec := &ExposureCalculator{}
	positions := map[string]float64{"A": 100, "B": -50}
	got := ec.Leverage(positions, 75)
	// gross = 150, equity = 75, leverage = 2.0
	if !approxEqual(got, 2.0) {
		t.Errorf("leverage: got %.4f want 2.0", got)
	}
}

func TestExposureCalculator_Leverage_ZeroEquity(t *testing.T) {
	ec := &ExposureCalculator{}
	got := ec.Leverage(map[string]float64{"A": 100}, 0)
	if got != 0 {
		t.Errorf("leverage zero equity: got %.4f want 0", got)
	}
}
