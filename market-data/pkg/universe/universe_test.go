package universe

import (
	"encoding/json"
	"os"
	"testing"
)

// -- UniverseManager tests --

func TestUniverseManager_DefaultUniverse_NotEmpty(t *testing.T) {
	m := NewUniverseManager()
	if m.Size() == 0 {
		t.Error("default universe is empty")
	}
}

func TestUniverseManager_ActiveSymbols(t *testing.T) {
	m := NewUniverseManager()
	active := m.ActiveSymbols()
	if len(active) == 0 {
		t.Error("no active symbols in default universe")
	}
	// All defaults are active, so count should equal size.
	if len(active) != m.Size() {
		t.Errorf("active count %d != size %d", len(active), m.Size())
	}
}

func TestUniverseManager_SymbolInfo_Found(t *testing.T) {
	m := NewUniverseManager()
	info, ok := m.SymbolInfo("AAPL")
	if !ok {
		t.Fatal("AAPL not found in default universe")
	}
	if info.Sector != "technology" {
		t.Errorf("AAPL sector: got %q want technology", info.Sector)
	}
	if info.AssetClass != "equity" {
		t.Errorf("AAPL asset class: got %q want equity", info.AssetClass)
	}
}

func TestUniverseManager_SymbolInfo_CaseInsensitive(t *testing.T) {
	m := NewUniverseManager()
	_, ok := m.SymbolInfo("aapl")
	if !ok {
		t.Error("lowercase aapl lookup failed")
	}
}

func TestUniverseManager_SymbolInfo_NotFound(t *testing.T) {
	m := NewUniverseManager()
	_, ok := m.SymbolInfo("ZZZZ")
	if ok {
		t.Error("expected not found for ZZZZ")
	}
}

func TestUniverseManager_FilterByADV(t *testing.T) {
	m := NewUniverseManager()
	// SPY has ADV30 = 25B, should be in results for minADV = 10B
	results := m.FilterByADV(10_000_000_000)
	found := false
	for _, s := range results {
		if s == "SPY" {
			found = true
			break
		}
	}
	if !found {
		t.Error("SPY should pass ADV filter at 10B")
	}
	// ARB has ADV 150M, should be excluded.
	for _, s := range results {
		if s == "ARB-USD" {
			t.Error("ARB-USD should not pass 10B ADV filter")
		}
	}
}

func TestUniverseManager_FilterBySector(t *testing.T) {
	m := NewUniverseManager()
	tech := m.FilterBySector("technology")
	if len(tech) == 0 {
		t.Error("expected technology symbols")
	}
	for _, s := range tech {
		info, _ := m.SymbolInfo(s)
		if info.Sector != "technology" {
			t.Errorf("symbol %s has sector %q not technology", s, info.Sector)
		}
	}
}

func TestUniverseManager_SetActive(t *testing.T) {
	m := NewUniverseManager()
	if err := m.SetActive("AAPL", false); err != nil {
		t.Fatal(err)
	}
	info, _ := m.SymbolInfo("AAPL")
	if info.IsActive {
		t.Error("AAPL should be inactive after SetActive(false)")
	}
	active := m.ActiveSymbols()
	for _, s := range active {
		if s == "AAPL" {
			t.Error("AAPL should not appear in ActiveSymbols")
		}
	}
}

func TestUniverseManager_LoadJSON(t *testing.T) {
	instruments := []InstrumentInfo{
		{Symbol: "TEST1", Name: "Test One", Exchange: "X", AssetClass: "equity",
			Sector: "tech", MarketCap: 1e9, ADV30: 1e7, IsActive: true},
		{Symbol: "TEST2", Name: "Test Two", Exchange: "X", AssetClass: "crypto",
			Sector: "defi", MarketCap: 5e8, ADV30: 5e6, IsActive: false},
	}

	f, err := os.CreateTemp("", "universe_*.json")
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(f.Name())

	if err := json.NewEncoder(f).Encode(instruments); err != nil {
		t.Fatal(err)
	}
	f.Close()

	m := NewUniverseManager()
	if err := m.LoadUniverse(f.Name()); err != nil {
		t.Fatal(err)
	}
	if m.Size() != 2 {
		t.Errorf("size after JSON load: got %d want 2", m.Size())
	}
	active := m.ActiveSymbols()
	if len(active) != 1 || active[0] != "TEST1" {
		t.Errorf("active after JSON load: got %v", active)
	}
}

// -- LiquidityRanker tests --

func TestLiquidityRanker_Rank_Order(t *testing.T) {
	r := NewLiquidityRanker(nil)
	symbols := []string{"A", "B", "C"}
	adv := map[string]float64{"A": 1_000_000, "B": 500_000, "C": 2_000_000}
	spread := map[string]float64{"A": 0.001, "B": 0.0005, "C": 0.002}

	ranked := r.Rank(symbols, adv, spread)
	if len(ranked) != 3 {
		t.Fatalf("expected 3 results, got %d", len(ranked))
	}
	// B has tightest spread (best) and decent ADV -- should rank highest.
	if ranked[0].Symbol != "B" {
		t.Errorf("top symbol: got %s want B", ranked[0].Symbol)
	}
}

func TestLiquidityRanker_TopN(t *testing.T) {
	r := NewLiquidityRanker(nil)
	symbols := []string{"A", "B", "C", "D"}
	adv := map[string]float64{"A": 100, "B": 200, "C": 300, "D": 400}
	spread := map[string]float64{"A": 0.01, "B": 0.008, "C": 0.005, "D": 0.002}
	r.Rank(symbols, adv, spread)

	top2 := r.TopN(2)
	if len(top2) != 2 {
		t.Errorf("TopN(2): got %d results", len(top2))
	}
}

func TestLiquidityRanker_TopN_ExceedsLength(t *testing.T) {
	r := NewLiquidityRanker(nil)
	symbols := []string{"A", "B"}
	adv := map[string]float64{"A": 100, "B": 200}
	spread := map[string]float64{"A": 0.01, "B": 0.005}
	r.Rank(symbols, adv, spread)
	top10 := r.TopN(10)
	if len(top10) != 2 {
		t.Errorf("TopN(10) with 2 symbols: got %d want 2", len(top10))
	}
}

func TestLiquidityRanker_IsLiquidEnough(t *testing.T) {
	m := NewUniverseManager()
	r := NewLiquidityRanker(m)

	// AAPL ADV30 = 8B, so 5% = 400M
	if !r.IsLiquidEnough("AAPL", 300_000_000) {
		t.Error("300M order in AAPL (ADV 8B) should be liquid enough")
	}
	if r.IsLiquidEnough("AAPL", 500_000_000) {
		t.Error("500M order exceeds 5% of AAPL ADV, should not be liquid enough")
	}
}

func TestLiquidityRanker_IsLiquidEnough_UnknownSymbol(t *testing.T) {
	m := NewUniverseManager()
	r := NewLiquidityRanker(m)
	if r.IsLiquidEnough("ZZZZ", 100) {
		t.Error("unknown symbol should not be liquid enough")
	}
}

func TestLiquidityRanker_ScoreFor(t *testing.T) {
	r := NewLiquidityRanker(nil)
	symbols := []string{"X", "Y"}
	r.Rank(symbols,
		map[string]float64{"X": 1000, "Y": 2000},
		map[string]float64{"X": 0.01, "Y": 0.005},
	)
	score, ok := r.ScoreFor("X")
	if !ok {
		t.Error("ScoreFor X: expected ok")
	}
	if score <= 0 || score > 1 {
		t.Errorf("score out of range: %v", score)
	}
}

func TestLiquidityRanker_EmptySymbols(t *testing.T) {
	r := NewLiquidityRanker(nil)
	result := r.Rank(nil, nil, nil)
	if result != nil {
		t.Error("empty symbols should return nil")
	}
}
