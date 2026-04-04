// Package duckdb — prebuilt_queries.go provides named, parameterised analytics
// queries that can be run via the DuckDB analytics layer.
package duckdb

import (
	"context"
	"fmt"
	"time"
)

// VWAPResult is the volume-weighted average price for a time window.
type VWAPResult struct {
	Symbol    string
	Date      time.Time
	VWAP      float64
	TotalVol  float64
}

// VWAP computes daily VWAP for a symbol.
func (a *Analytics) VWAP(ctx context.Context, symbol string, from, to time.Time) ([]VWAPResult, error) {
	q := fmt.Sprintf(`
SELECT
  date_trunc('day', timestamp) AS day,
  symbol,
  SUM(close * volume) / NULLIF(SUM(volume), 0) AS vwap,
  SUM(volume) AS total_volume
FROM bars
WHERE symbol = '%s'
  AND timestamp >= TIMESTAMP '%s'
  AND timestamp <= TIMESTAMP '%s'
GROUP BY 1, 2
ORDER BY 1
`,
		escapeSQLString(symbol),
		from.Format("2006-01-02"),
		to.Format("2006-01-02"),
	)

	rows, err := a.RunQuery(ctx, q)
	if err != nil {
		return nil, err
	}

	var out []VWAPResult
	for _, row := range rows {
		day, _ := parseTime(fmt.Sprintf("%v", row["day"]))
		out = append(out, VWAPResult{
			Symbol:   symbol,
			Date:     day,
			VWAP:     toFloat64(row["vwap"]),
			TotalVol: toFloat64(row["total_volume"]),
		})
	}
	return out, nil
}

// RSIResult holds RSI values over time.
type RSIResult struct {
	Symbol    string
	Timestamp time.Time
	RSI       float64
}

// RSI computes the Relative Strength Index using a DuckDB window function.
func (a *Analytics) RSI(ctx context.Context, symbol string, period int, from, to time.Time) ([]RSIResult, error) {
	q := fmt.Sprintf(`
WITH base AS (
  SELECT
    date_trunc('day', timestamp) AS day,
    symbol,
    last(close ORDER BY timestamp) AS close_price
  FROM bars
  WHERE symbol = '%s'
    AND timestamp >= TIMESTAMP '%s'
    AND timestamp <= TIMESTAMP '%s'
  GROUP BY 1, 2
),
changes AS (
  SELECT
    day,
    symbol,
    close_price,
    close_price - LAG(close_price) OVER (PARTITION BY symbol ORDER BY day) AS change
  FROM base
),
gains_losses AS (
  SELECT
    day,
    symbol,
    CASE WHEN change > 0 THEN change ELSE 0 END AS gain,
    CASE WHEN change < 0 THEN -change ELSE 0 END AS loss
  FROM changes
  WHERE change IS NOT NULL
),
avg_gl AS (
  SELECT
    day,
    symbol,
    AVG(gain) OVER (PARTITION BY symbol ORDER BY day ROWS BETWEEN %d PRECEDING AND CURRENT ROW) AS avg_gain,
    AVG(loss) OVER (PARTITION BY symbol ORDER BY day ROWS BETWEEN %d PRECEDING AND CURRENT ROW) AS avg_loss
  FROM gains_losses
)
SELECT
  day,
  symbol,
  100 - (100 / (1 + avg_gain / NULLIF(avg_loss, 0))) AS rsi
FROM avg_gl
WHERE avg_loss IS NOT NULL
ORDER BY day
`,
		escapeSQLString(symbol),
		from.Format("2006-01-02"),
		to.Format("2006-01-02"),
		period-1, period-1,
	)

	rows, err := a.RunQuery(ctx, q)
	if err != nil {
		return nil, err
	}

	var out []RSIResult
	for _, row := range rows {
		day, _ := parseTime(fmt.Sprintf("%v", row["day"]))
		out = append(out, RSIResult{
			Symbol:    symbol,
			Timestamp: day,
			RSI:       toFloat64(row["rsi"]),
		})
	}
	return out, nil
}

// BollingerBands computes Bollinger Bands for a symbol.
type BollingerResult struct {
	Symbol    string
	Timestamp time.Time
	Middle    float64
	Upper     float64
	Lower     float64
	Close     float64
	BWidth    float64 // band width as pct of middle
}

// BollingerBands computes Bollinger Bands (period, numStdDev).
func (a *Analytics) BollingerBands(ctx context.Context, symbol string, period int, numStdDev float64, from, to time.Time) ([]BollingerResult, error) {
	q := fmt.Sprintf(`
WITH base AS (
  SELECT
    date_trunc('day', timestamp) AS day,
    symbol,
    last(close ORDER BY timestamp) AS close_price
  FROM bars
  WHERE symbol = '%s'
    AND timestamp >= TIMESTAMP '%s'
    AND timestamp <= TIMESTAMP '%s'
  GROUP BY 1, 2
),
bb AS (
  SELECT
    day,
    symbol,
    close_price,
    AVG(close_price) OVER (PARTITION BY symbol ORDER BY day ROWS BETWEEN %d PRECEDING AND CURRENT ROW) AS middle,
    STDDEV(close_price) OVER (PARTITION BY symbol ORDER BY day ROWS BETWEEN %d PRECEDING AND CURRENT ROW) AS std
  FROM base
)
SELECT
  day,
  symbol,
  close_price,
  middle,
  middle + %f * std AS upper_band,
  middle - %f * std AS lower_band,
  CASE WHEN middle > 0 THEN (2 * %f * std) / middle * 100 ELSE 0 END AS band_width
FROM bb
WHERE std IS NOT NULL AND std > 0
ORDER BY day
`,
		escapeSQLString(symbol),
		from.Format("2006-01-02"),
		to.Format("2006-01-02"),
		period-1, period-1,
		numStdDev, numStdDev, numStdDev,
	)

	rows, err := a.RunQuery(ctx, q)
	if err != nil {
		return nil, err
	}

	var out []BollingerResult
	for _, row := range rows {
		day, _ := parseTime(fmt.Sprintf("%v", row["day"]))
		out = append(out, BollingerResult{
			Symbol:    symbol,
			Timestamp: day,
			Close:     toFloat64(row["close_price"]),
			Middle:    toFloat64(row["middle"]),
			Upper:     toFloat64(row["upper_band"]),
			Lower:     toFloat64(row["lower_band"]),
			BWidth:    toFloat64(row["band_width"]),
		})
	}
	return out, nil
}

// DrawdownSeries computes a running drawdown series.
type DrawdownPoint struct {
	Symbol    string
	Timestamp time.Time
	Equity    float64
	Peak      float64
	Drawdown  float64 // negative, as fraction
}

// DrawdownSeries computes the equity drawdown curve.
func (a *Analytics) DrawdownSeries(ctx context.Context, symbol string, from, to time.Time) ([]DrawdownPoint, error) {
	q := fmt.Sprintf(`
WITH base AS (
  SELECT
    date_trunc('day', timestamp) AS day,
    symbol,
    last(close ORDER BY timestamp) AS close_price
  FROM bars
  WHERE symbol = '%s'
    AND timestamp >= TIMESTAMP '%s'
    AND timestamp <= TIMESTAMP '%s'
  GROUP BY 1, 2
),
cumulative AS (
  SELECT
    day,
    symbol,
    close_price,
    MAX(close_price) OVER (PARTITION BY symbol ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS peak
  FROM base
)
SELECT
  day,
  symbol,
  close_price,
  peak,
  (close_price - peak) / NULLIF(peak, 0) AS drawdown
FROM cumulative
ORDER BY day
`,
		escapeSQLString(symbol),
		from.Format("2006-01-02"),
		to.Format("2006-01-02"),
	)

	rows, err := a.RunQuery(ctx, q)
	if err != nil {
		return nil, err
	}

	var out []DrawdownPoint
	for _, row := range rows {
		day, _ := parseTime(fmt.Sprintf("%v", row["day"]))
		out = append(out, DrawdownPoint{
			Symbol:    symbol,
			Timestamp: day,
			Equity:    toFloat64(row["close_price"]),
			Peak:      toFloat64(row["peak"]),
			Drawdown:  toFloat64(row["drawdown"]),
		})
	}
	return out, nil
}

// VolumeSummary holds daily volume statistics.
type VolumeSummary struct {
	Symbol   string
	Date     time.Time
	Volume   float64
	AvgVol20 float64
	VolRatio float64
}

// VolumeSummaries returns daily volume with 20-day rolling average.
func (a *Analytics) VolumeSummaries(ctx context.Context, symbol string, from, to time.Time) ([]VolumeSummary, error) {
	q := fmt.Sprintf(`
WITH base AS (
  SELECT
    date_trunc('day', timestamp) AS day,
    symbol,
    SUM(volume) AS daily_volume
  FROM bars
  WHERE symbol = '%s'
    AND timestamp >= TIMESTAMP '%s'
    AND timestamp <= TIMESTAMP '%s'
  GROUP BY 1, 2
),
rolling AS (
  SELECT
    day,
    symbol,
    daily_volume,
    AVG(daily_volume) OVER (PARTITION BY symbol ORDER BY day ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS avg_vol_20
  FROM base
)
SELECT
  day,
  symbol,
  daily_volume,
  avg_vol_20,
  daily_volume / NULLIF(avg_vol_20, 0) AS vol_ratio
FROM rolling
ORDER BY day
`,
		escapeSQLString(symbol),
		from.Format("2006-01-02"),
		to.Format("2006-01-02"),
	)

	rows, err := a.RunQuery(ctx, q)
	if err != nil {
		return nil, err
	}

	var out []VolumeSummary
	for _, row := range rows {
		day, _ := parseTime(fmt.Sprintf("%v", row["day"]))
		out = append(out, VolumeSummary{
			Symbol:   symbol,
			Date:     day,
			Volume:   toFloat64(row["daily_volume"]),
			AvgVol20: toFloat64(row["avg_vol_20"]),
			VolRatio: toFloat64(row["vol_ratio"]),
		})
	}
	return out, nil
}

// SymbolStats returns summary statistics for a symbol over a time range.
type SymbolStats struct {
	Symbol       string
	From         time.Time
	To           time.Time
	TotalReturn  float64
	AnnReturn    float64
	Volatility   float64
	Sharpe       float64
	MaxDrawdown  float64
	AvgDailyVol  float64
	TradingDays  int
}

// SymbolStatistics computes comprehensive statistics for a symbol.
func (a *Analytics) SymbolStatistics(ctx context.Context, symbol string, from, to time.Time) (*SymbolStats, error) {
	rets, err := a.DailyReturns(ctx, symbol, from, to)
	if err != nil {
		return nil, err
	}
	if len(rets) == 0 {
		return &SymbolStats{Symbol: symbol, From: from, To: to}, nil
	}

	// Compute statistics from returns.
	n := float64(len(rets))
	var sum, sumSq float64
	for _, r := range rets {
		sum += r.Return
		sumSq += r.Return * r.Return
	}
	mean := sum / n
	variance := (sumSq - sum*sum/n) / (n - 1)

	sqrtFn := func(v float64) float64 {
		if v <= 0 {
			return 0
		}
		z := v
		for i := 0; i < 50; i++ {
			z = (z + v/z) / 2
		}
		return z
	}

	vol := sqrtFn(variance) * sqrtFn(252)
	annReturn := mean * 252
	sharpe := 0.0
	if vol > 0 {
		sharpe = annReturn / vol
	}

	// Total return.
	equity := 1.0
	for _, r := range rets {
		equity *= (1 + r.Return)
	}
	totalReturn := equity - 1

	// Max drawdown.
	peak := 1.0
	eq := 1.0
	maxDD := 0.0
	for _, r := range rets {
		eq *= (1 + r.Return)
		if eq > peak {
			peak = eq
		}
		dd := (peak - eq) / peak
		if dd > maxDD {
			maxDD = dd
		}
	}

	vols, _ := a.VolumeSummaries(ctx, symbol, from, to)
	var totalVol float64
	for _, v := range vols {
		totalVol += v.Volume
	}
	avgVol := 0.0
	if len(vols) > 0 {
		avgVol = totalVol / float64(len(vols))
	}

	return &SymbolStats{
		Symbol:      symbol,
		From:        from,
		To:          to,
		TotalReturn: totalReturn,
		AnnReturn:   annReturn,
		Volatility:  vol,
		Sharpe:      sharpe,
		MaxDrawdown: maxDD,
		AvgDailyVol: avgVol,
		TradingDays: len(rets),
	}, nil
}
