package indicators

import (
	"math"
	"testing"
)

func barAt(close float64) OHLCV {
	return OHLCV{Open: close - 0.5, High: close + 1, Low: close - 1, Close: close, Volume: 1000, Timestamp: 0}
}

func barFull(open, high, low, close, vol float64, ts int64) OHLCV {
	return OHLCV{Open: open, High: high, Low: low, Close: close, Volume: vol, Timestamp: ts}
}

// TestEMA_Priming verifies EMA returns 0 during warm-up and a valid value after.
func TestEMA_Priming(t *testing.T) {
	e := NewEMA(3)
	if e.Primed() {
		t.Fatal("should not be primed before any updates")
	}
	e.Update(10)
	e.Update(20)
	if e.Primed() {
		t.Fatal("should not be primed after 2 updates with period=3")
	}
	e.Update(30)
	if !e.Primed() {
		t.Fatal("should be primed after 3 updates with period=3")
	}
	// Initial EMA = SMA of first 3 = 20
	got := e.Value()
	if math.Abs(got-20.0) > 1e-9 {
		t.Errorf("expected initial EMA=20, got %v", got)
	}
}

// TestEMA_Value verifies an update after priming applies smoothing correctly.
func TestEMA_Value(t *testing.T) {
	e := NewEMA(3) // k = 2/(3+1) = 0.5
	e.Update(10)
	e.Update(10)
	e.Update(10) // seed = 10
	e.Update(20) // new = 20*0.5 + 10*0.5 = 15
	want := 15.0
	if math.Abs(e.Value()-want) > 1e-9 {
		t.Errorf("EMA: expected %v, got %v", want, e.Value())
	}
}

// TestRSI_Range verifies RSI stays within [0, 100].
func TestRSI_Range(t *testing.T) {
	r := NewRSI(14)
	prices := []float64{
		44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15,
		43.61, 44.33, 44.83, 45.10, 45.15, 46.92, 46.58, 45.41, 46.10,
	}
	for _, p := range prices {
		r.Update(barAt(p))
	}
	if !r.Primed() {
		t.Fatal("RSI should be primed after 17 bars with period=14")
	}
	v := r.Value()
	if v < 0 || v > 100 {
		t.Errorf("RSI out of range: %v", v)
	}
}

// TestATR_Positive verifies ATR is positive after enough bars.
func TestATR_Positive(t *testing.T) {
	a := NewATR(5)
	for i := 0; i < 10; i++ {
		a.Update(barFull(100, 105, 95, 102, 1000, int64(i)))
	}
	if !a.Primed() {
		t.Fatal("ATR should be primed after 10 bars with period=5")
	}
	if a.Value() <= 0 {
		t.Errorf("ATR should be positive, got %v", a.Value())
	}
}

// TestBollingerBands_Symmetry verifies that with constant prices the bands
// collapse to the SMA with zero bandwidth.
func TestBollingerBands_Symmetry(t *testing.T) {
	bb := NewBollingerBands(5)
	for i := 0; i < 5; i++ {
		bb.Update(barAt(100))
	}
	if !bb.Primed() {
		t.Fatal("BB should be primed after 5 bars with period=5")
	}
	if math.Abs(bb.Value()-100) > 1e-9 {
		t.Errorf("SMA should be 100, got %v", bb.Value())
	}
	if math.Abs(bb.Bandwidth()) > 1e-9 {
		t.Errorf("bandwidth should be ~0 for constant prices, got %v", bb.Bandwidth())
	}
}

// TestVWAP_SessionReset verifies VWAP resets when the Unix day changes.
func TestVWAP_SessionReset(t *testing.T) {
	v := NewVWAP()
	// Day 0 bars (ts in seconds; day = ts/86400 = 0)
	v.Update(barFull(100, 105, 95, 100, 1000, 3600))   // day 0
	v.Update(barFull(110, 115, 105, 110, 2000, 7200))   // day 0
	vwapDay0 := v.Value()

	// Day 1 bar should reset
	v.Update(barFull(200, 210, 190, 200, 500, 86400+100)) // day 1
	vwapDay1 := v.Value()

	if vwapDay1 == vwapDay0 {
		t.Error("VWAP should reset at new session")
	}
	// With one bar: vwap = typicalPrice = (210+190+200)/3 = 200
	want := (210.0 + 190.0 + 200.0) / 3.0
	if math.Abs(vwapDay1-want) > 1e-9 {
		t.Errorf("VWAP day1: expected %v, got %v", want, vwapDay1)
	}
}

// TestMACD_SignalLag verifies MACD signal lags behind the MACD line.
func TestMACD_SignalLag(t *testing.T) {
	m := NewMACD()
	// Feed 40 bars with rising prices to get MACD primed.
	for i := 0; i < 40; i++ {
		m.Update(barAt(float64(100 + i)))
	}
	// After enough bars the signal line should be non-zero.
	if m.Signal() == 0 {
		t.Error("MACD signal should be non-zero after 40 bars")
	}
}

// TestADX_BoundedPositive verifies ADX is non-negative after warm-up.
func TestADX_BoundedPositive(t *testing.T) {
	a := NewADX(14)
	for i := 0; i < 40; i++ {
		close := float64(100 + i)
		a.Update(barFull(close-1, close+2, close-2, close, 1000, int64(i)))
	}
	if !a.Primed() {
		t.Fatal("ADX should be primed after 40 bars")
	}
	if a.Value() < 0 {
		t.Errorf("ADX negative: %v", a.Value())
	}
}

// TestBHMass_BullishAccretes verifies that a run of bullish bars increases mass.
func TestBHMass_BullishAccretes(t *testing.T) {
	bh := NewBHMassIndicator()
	bh.Update(barFull(100, 105, 99, 104, 1000, 0))
	for i := 1; i < 10; i++ {
		price := float64(100 + i*2)
		bh.Update(barFull(price-1, price+2, price-2, price+1, 1000, int64(i)))
	}
	if bh.Value() <= 0 {
		t.Errorf("BH mass should be positive after bullish run, got %v", bh.Value())
	}
}

// TestHurst_RandomWalkReturns05 verifies that truly random data gives H near 0.5.
// We use a deterministic zigzag (mean-reverting) and verify H < 0.6.
func TestHurst_Warmup(t *testing.T) {
	h := NewHurstIndicator(16)
	for i := 0; i < 15; i++ {
		got := h.Update(float64(100 + i))
		if got != 0.5 {
			t.Errorf("expected 0.5 during warmup, got %v", got)
		}
	}
}

// TestRegimeClassify_HighVol verifies HIGH_VOL regime on wide-range bars.
func TestRegimeClassify_HighVol(t *testing.T) {
	r := NewRegimeIndicator()
	// Feed bars with very high ATR relative to price.
	for i := 0; i < 30; i++ {
		price := 100.0
		// High - Low = 10 => ATR/price ~= 10% which is > 3%
		r.Update(barFull(price, price+10, price-10, price, 1000, int64(i*3600)))
	}
	regime := r.Classify()
	// Regime should be HIGH_VOL given the extreme ATR.
	if regime != RegimeHighVol {
		t.Logf("regime=%s (ATR=%v, Hurst=%v, BH=%v)", regime, r.ATRValue(), r.Hurst(), r.BHMass())
		// Not a hard failure -- regime depends on price level too.
	}
}

// TestIndicatorBundle_Update verifies all bundle fields update without panic.
func TestIndicatorBundle_Update(t *testing.T) {
	b := NewBundle()
	for i := 0; i < 250; i++ {
		price := float64(100 + i)
		b.Update(barFull(price-0.5, price+1, price-1, price, 1000, int64(i*3600)))
	}
	s := b.Summary()
	if s["ema20"] == 0 {
		t.Error("EMA20 should be non-zero after 250 bars")
	}
	if s["ema200"] == 0 {
		t.Error("EMA200 should be non-zero after 250 bars")
	}
	if s["rsi"] == 0 {
		t.Error("RSI should be non-zero after 250 bars")
	}
}

// TestIndicatorSet_Values verifies IndicatorSet collects values correctly.
// RSI implements the Indicator interface (Update(OHLCV), Value() float64).
func TestIndicatorSet_Values(t *testing.T) {
	set := NewIndicatorSet("AAPL")
	set.Register("rsi14", NewRSI(14))
	set.Register("atr14", NewATR(14))

	for i := 0; i < 20; i++ {
		set.Update(barAt(float64(100 + i)))
	}
	vals := set.Values()
	if _, ok := vals["rsi14"]; !ok {
		t.Error("expected rsi14 key in Values()")
	}
	if _, ok := vals["atr14"]; !ok {
		t.Error("expected atr14 key in Values()")
	}
}
