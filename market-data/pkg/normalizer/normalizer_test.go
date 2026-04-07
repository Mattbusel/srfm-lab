package normalizer

import (
	"math"
	"os"
	"testing"
	"time"
)

func approxEq(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// -- PriceNormalizer tests --

func tempDB(t *testing.T) string {
	t.Helper()
	f, err := os.CreateTemp("", "norm_test_*.db")
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	t.Cleanup(func() { os.Remove(f.Name()) })
	return f.Name()
}

func TestPriceNormalizer_USD_NoAdjustment(t *testing.T) {
	n, err := NewPriceNormalizer(tempDB(t), 4)
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()

	raw := RawBar{
		Symbol: "AAPL", Open: 150, High: 155, Low: 149, Close: 153,
		Volume: 1e6, Timestamp: 1_000_000, Currency: "USD", Source: "NYSE",
	}
	nb, err := n.Normalize(raw, "NYSE")
	if err != nil {
		t.Fatal(err)
	}
	if nb.OHLCV[3] != 153.0 {
		t.Errorf("close: got %v want 153", nb.OHLCV[3])
	}
	if nb.AdjFactor != 1.0 {
		t.Errorf("adj factor: got %v want 1.0", nb.AdjFactor)
	}
}

func TestPriceNormalizer_USDT_TreatedAsUSD(t *testing.T) {
	n, err := NewPriceNormalizer(tempDB(t), 2)
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()

	raw := RawBar{
		Symbol: "BTC-USD", Open: 60000, High: 62000, Low: 59000, Close: 61000,
		Volume: 500, Timestamp: 2_000_000, Currency: "USDT", Source: "binance",
	}
	nb, err := n.Normalize(raw, "binance")
	if err != nil {
		t.Fatal(err)
	}
	// USDT rate = 1.0, so close should be 61000.00
	if nb.OHLCV[3] != 61000.0 {
		t.Errorf("BTC close: got %v want 61000", nb.OHLCV[3])
	}
}

func TestPriceNormalizer_Adjustment_Applied(t *testing.T) {
	n, err := NewPriceNormalizer(tempDB(t), 4)
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()

	exDate := time.UnixMilli(500_000)
	if err := n.RegisterAdjustment("TSLA", 0.5, exDate); err != nil { // 2:1 split => factor 0.5
		t.Fatal(err)
	}

	// Bar before the ex date should get the adjustment.
	raw := RawBar{
		Symbol: "TSLA", Open: 400, High: 420, Low: 390, Close: 410,
		Volume: 1e6, Timestamp: 400_000, Currency: "USD",
	}
	nb, err := n.Normalize(raw, "nasdaq")
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(nb.OHLCV[3], 410*0.5, 1e-4) {
		t.Errorf("adjusted close: got %v want %v", nb.OHLCV[3], 410*0.5)
	}
}

func TestPriceNormalizer_Adjustment_NotAppliedAfterExDate(t *testing.T) {
	n, err := NewPriceNormalizer(tempDB(t), 4)
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()

	exDate := time.UnixMilli(500_000)
	n.RegisterAdjustment("TSLA", 0.5, exDate) //nolint:errcheck

	// Bar after the ex date -- the factor query returns no rows -> 1.0.
	raw := RawBar{
		Symbol: "TSLA", Close: 210, Timestamp: 600_000, Currency: "USD",
	}
	nb, err := n.Normalize(raw, "nasdaq")
	if err != nil {
		t.Fatal(err)
	}
	if !approxEq(nb.OHLCV[3], 210.0, 1e-4) {
		t.Errorf("post-exdate close: got %v want 210", nb.OHLCV[3])
	}
}

func TestPriceNormalizer_UnknownCurrency_Error(t *testing.T) {
	n, err := NewPriceNormalizer(tempDB(t), 2)
	if err != nil {
		t.Fatal(err)
	}
	defer n.Close()

	raw := RawBar{Symbol: "X", Close: 100, Currency: "XYZ", Timestamp: 1}
	_, err = n.Normalize(raw, "test")
	if err == nil {
		t.Error("expected error for unknown currency, got nil")
	}
}

func TestRoundTo(t *testing.T) {
	cases := []struct {
		v    float64
		d    int
		want float64
	}{
		{3.14159, 2, 3.14},
		{3.145, 2, 3.15},
		{100.0, 0, 100},
		{0.00049, 3, 0},
		{0.00050, 3, 0.001},
	}
	for _, tc := range cases {
		got := roundTo(tc.v, tc.d)
		if !approxEq(got, tc.want, 1e-9) {
			t.Errorf("roundTo(%v, %d): got %v want %v", tc.v, tc.d, got, tc.want)
		}
	}
}

// -- CorporateActionHandler tests --

func TestCorporateAction_AdjustHistory_Split(t *testing.T) {
	h, err := NewCorporateActionHandler(tempDB(t))
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	exDate := time.UnixMilli(1_000_000)
	actions := []CorporateAction{
		{Symbol: "NVDA", Type: ActionSplit, ExDate: exDate, Factor: 10.0}, // 10:1 split
	}

	bars := []NormalizedBar{
		{Symbol: "NVDA", OHLCV: [5]float64{1000, 1100, 950, 1050, 5000}, Timestamp: 500_000, AdjFactor: 1.0},
		{Symbol: "NVDA", OHLCV: [5]float64{100, 110, 95, 105, 50000}, Timestamp: 1_500_000, AdjFactor: 1.0},
	}

	adjusted := h.AdjustHistory(bars, actions)

	// Bar before ex_date should have prices divided by 10, volume multiplied by 10.
	if !approxEq(adjusted[0].OHLCV[3], 105.0, 1e-6) {
		t.Errorf("pre-split close: got %v want 105", adjusted[0].OHLCV[3])
	}
	if !approxEq(adjusted[0].OHLCV[4], 50000.0, 1e-6) {
		t.Errorf("pre-split volume: got %v want 50000", adjusted[0].OHLCV[4])
	}

	// Bar after ex_date is not adjusted.
	if !approxEq(adjusted[1].OHLCV[3], 105.0, 1e-6) {
		t.Errorf("post-split close: got %v want 105", adjusted[1].OHLCV[3])
	}
}

func TestCorporateAction_AdjustHistory_Dividend(t *testing.T) {
	h, err := NewCorporateActionHandler(tempDB(t))
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	exDate := time.UnixMilli(1_000_000)
	actions := []CorporateAction{
		{Symbol: "AAPL", Type: ActionDividend, ExDate: exDate, Factor: 0.25},
	}

	bars := []NormalizedBar{
		{Symbol: "AAPL", OHLCV: [5]float64{200, 205, 198, 202, 1e6}, Timestamp: 500_000},
	}
	adjusted := h.AdjustHistory(bars, actions)
	want := 202 - 0.25
	if !approxEq(adjusted[0].OHLCV[3], want, 1e-6) {
		t.Errorf("dividend adjusted close: got %v want %v", adjusted[0].OHLCV[3], want)
	}
}

func TestCorporateAction_PendingActions(t *testing.T) {
	h, err := NewCorporateActionHandler(tempDB(t))
	if err != nil {
		t.Fatal(err)
	}
	defer h.Close()

	future := time.Now().Add(24 * time.Hour)
	past := time.Now().Add(-24 * time.Hour)

	h.AddAction(CorporateAction{Symbol: "A", Type: ActionSplit, ExDate: future, Factor: 2}) //nolint
	h.AddAction(CorporateAction{Symbol: "B", Type: ActionDividend, ExDate: past, Factor: 1}) //nolint

	pending, err := h.PendingActions()
	if err != nil {
		t.Fatal(err)
	}
	if len(pending) != 1 {
		t.Errorf("pending actions: got %d want 1", len(pending))
	}
	if pending[0].Symbol != "A" {
		t.Errorf("pending symbol: got %s want A", pending[0].Symbol)
	}
}
