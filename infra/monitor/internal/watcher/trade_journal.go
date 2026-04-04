package watcher

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"
)

// TradeDirection is buy or sell.
type TradeDirection string

const (
	DirectionBuy  TradeDirection = "buy"
	DirectionSell TradeDirection = "sell"
)

// Trade represents a completed round-trip trade.
type Trade struct {
	ID         string         `json:"id"`
	Symbol     string         `json:"symbol"`
	Direction  TradeDirection `json:"direction"`
	EntryTime  time.Time      `json:"entry_time"`
	ExitTime   time.Time      `json:"exit_time"`
	EntryPrice float64        `json:"entry_price"`
	ExitPrice  float64        `json:"exit_price"`
	Quantity   float64        `json:"quantity"`
	PnL        float64        `json:"pnl"`
	ReturnPct  float64        `json:"return_pct"`
	Commission float64        `json:"commission"`
	Slippage   float64        `json:"slippage"`
	Tag        string         `json:"tag,omitempty"`
}

// TradeJournal records and analyses completed trades.
type TradeJournal struct {
	mu     sync.RWMutex
	trades []Trade
	file   string
}

// NewTradeJournal creates a TradeJournal, optionally loading from a CSV file.
func NewTradeJournal(csvPath string) (*TradeJournal, error) {
	tj := &TradeJournal{file: csvPath}
	if csvPath != "" {
		if err := tj.loadCSV(csvPath); err != nil && !os.IsNotExist(err) {
			return nil, fmt.Errorf("load journal: %w", err)
		}
	}
	return tj, nil
}

// AddTrade appends a completed trade and optionally persists it.
func (tj *TradeJournal) AddTrade(t Trade) error {
	tj.mu.Lock()
	tj.trades = append(tj.trades, t)
	tj.mu.Unlock()
	if tj.file != "" {
		return tj.appendCSV(t)
	}
	return nil
}

// Trades returns a copy of all trades.
func (tj *TradeJournal) Trades() []Trade {
	tj.mu.RLock()
	defer tj.mu.RUnlock()
	out := make([]Trade, len(tj.trades))
	copy(out, tj.trades)
	return out
}

// TradesBySymbol returns trades filtered by symbol.
func (tj *TradeJournal) TradesBySymbol(symbol string) []Trade {
	tj.mu.RLock()
	defer tj.mu.RUnlock()
	var out []Trade
	for _, t := range tj.trades {
		if t.Symbol == symbol {
			out = append(out, t)
		}
	}
	return out
}

// TradesByDateRange returns trades with exit time in [from, to].
func (tj *TradeJournal) TradesByDateRange(from, to time.Time) []Trade {
	tj.mu.RLock()
	defer tj.mu.RUnlock()
	var out []Trade
	for _, t := range tj.trades {
		if !t.ExitTime.Before(from) && !t.ExitTime.After(to) {
			out = append(out, t)
		}
	}
	return out
}

// JournalStats holds aggregate statistics for a set of trades.
type JournalStats struct {
	TotalTrades    int     `json:"total_trades"`
	WinRate        float64 `json:"win_rate"`
	TotalPnL       float64 `json:"total_pnl"`
	AvgPnL         float64 `json:"avg_pnl"`
	AvgWin         float64 `json:"avg_win"`
	AvgLoss        float64 `json:"avg_loss"`
	ProfitFactor   float64 `json:"profit_factor"`
	MaxWin         float64 `json:"max_win"`
	MaxLoss        float64 `json:"max_loss"`
	AvgHoldSeconds float64 `json:"avg_hold_seconds"`
	SharpeRatio    float64 `json:"sharpe_ratio"`
	ExpectedValue  float64 `json:"expected_value"`
	LargestDD      float64 `json:"largest_drawdown"`
}

// Stats computes aggregate statistics for a slice of trades.
func Stats(trades []Trade) JournalStats {
	if len(trades) == 0 {
		return JournalStats{}
	}

	wins, losses := 0, 0
	totalPnL := 0.0
	sumWin, sumLoss := 0.0, 0.0
	maxWin, maxLoss := 0.0, 0.0
	holdSecs := 0.0
	pnls := make([]float64, len(trades))

	for i, t := range trades {
		pnls[i] = t.PnL
		totalPnL += t.PnL
		holdSecs += t.ExitTime.Sub(t.EntryTime).Seconds()
		if t.PnL >= 0 {
			wins++
			sumWin += t.PnL
			if t.PnL > maxWin {
				maxWin = t.PnL
			}
		} else {
			losses++
			sumLoss += math.Abs(t.PnL)
			if math.Abs(t.PnL) > maxLoss {
				maxLoss = math.Abs(t.PnL)
			}
		}
	}

	n := float64(len(trades))
	winRate := float64(wins) / n

	avgWin := 0.0
	if wins > 0 {
		avgWin = sumWin / float64(wins)
	}
	avgLoss := 0.0
	if losses > 0 {
		avgLoss = sumLoss / float64(losses)
	}
	pf := 0.0
	if sumLoss > 0 {
		pf = sumWin / sumLoss
	}

	// EV = winRate * avgWin - lossRate * avgLoss
	ev := winRate*avgWin - (1-winRate)*avgLoss

	// Sharpe on trade-level PnL.
	mean := totalPnL / n
	variance := 0.0
	for _, p := range pnls {
		d := p - mean
		variance += d * d
	}
	stddev := 0.0
	if n > 1 {
		stddev = math.Sqrt(variance / (n - 1))
	}
	sharpe := 0.0
	if stddev > 0 {
		sharpe = mean / stddev * math.Sqrt(n)
	}

	// Max drawdown on cumulative PnL.
	peak := 0.0
	cum := 0.0
	maxDD := 0.0
	for _, p := range pnls {
		cum += p
		if cum > peak {
			peak = cum
		}
		dd := peak - cum
		if dd > maxDD {
			maxDD = dd
		}
	}

	return JournalStats{
		TotalTrades:    len(trades),
		WinRate:        winRate,
		TotalPnL:       totalPnL,
		AvgPnL:         mean,
		AvgWin:         avgWin,
		AvgLoss:        avgLoss,
		ProfitFactor:   pf,
		MaxWin:         maxWin,
		MaxLoss:        maxLoss,
		AvgHoldSeconds: holdSecs / n,
		SharpeRatio:    sharpe,
		ExpectedValue:  ev,
		LargestDD:      maxDD,
	}
}

// SymbolBreakdown returns per-symbol stats.
func SymbolBreakdown(trades []Trade) map[string]JournalStats {
	bySymbol := make(map[string][]Trade)
	for _, t := range trades {
		bySymbol[t.Symbol] = append(bySymbol[t.Symbol], t)
	}
	result := make(map[string]JournalStats, len(bySymbol))
	for sym, ts := range bySymbol {
		result[sym] = Stats(ts)
	}
	return result
}

// MonthlyBreakdown returns monthly PnL grouped by YYYY-MM.
func MonthlyBreakdown(trades []Trade) map[string]float64 {
	out := make(map[string]float64)
	for _, t := range trades {
		key := t.ExitTime.Format("2006-01")
		out[key] += t.PnL
	}
	return out
}

// SortedSymbols returns symbol names sorted by total PnL descending.
func SortedSymbols(breakdown map[string]JournalStats) []string {
	syms := make([]string, 0, len(breakdown))
	for s := range breakdown {
		syms = append(syms, s)
	}
	sort.Slice(syms, func(i, j int) bool {
		return breakdown[syms[i]].TotalPnL > breakdown[syms[j]].TotalPnL
	})
	return syms
}

// ExportJSON writes all trades to a JSON file.
func (tj *TradeJournal) ExportJSON(path string) error {
	tj.mu.RLock()
	trades := make([]Trade, len(tj.trades))
	copy(trades, tj.trades)
	tj.mu.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(trades)
}

// csvHeader returns the CSV column names.
func csvHeader() []string {
	return []string{"id", "symbol", "direction", "entry_time", "exit_time",
		"entry_price", "exit_price", "quantity", "pnl", "return_pct",
		"commission", "slippage", "tag"}
}

func tradeToRecord(t Trade) []string {
	return []string{
		t.ID,
		t.Symbol,
		string(t.Direction),
		t.EntryTime.Format(time.RFC3339),
		t.ExitTime.Format(time.RFC3339),
		strconv.FormatFloat(t.EntryPrice, 'f', 6, 64),
		strconv.FormatFloat(t.ExitPrice, 'f', 6, 64),
		strconv.FormatFloat(t.Quantity, 'f', 6, 64),
		strconv.FormatFloat(t.PnL, 'f', 6, 64),
		strconv.FormatFloat(t.ReturnPct, 'f', 6, 64),
		strconv.FormatFloat(t.Commission, 'f', 6, 64),
		strconv.FormatFloat(t.Slippage, 'f', 6, 64),
		t.Tag,
	}
}

func (tj *TradeJournal) appendCSV(t Trade) error {
	f, err := os.OpenFile(tj.file, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return err
	}
	defer f.Close()

	// Write header if new file.
	info, _ := f.Stat()
	w := csv.NewWriter(f)
	if info.Size() == 0 {
		if err := w.Write(csvHeader()); err != nil {
			return err
		}
	}
	if err := w.Write(tradeToRecord(t)); err != nil {
		return err
	}
	w.Flush()
	return w.Error()
}

func (tj *TradeJournal) loadCSV(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return err
	}
	if len(records) < 2 {
		return nil
	}
	for _, rec := range records[1:] {
		if len(rec) < 13 {
			continue
		}
		t := Trade{
			ID:        rec[0],
			Symbol:    rec[1],
			Direction: TradeDirection(rec[2]),
			Tag:       rec[12],
		}
		t.EntryTime, _ = time.Parse(time.RFC3339, rec[3])
		t.ExitTime, _ = time.Parse(time.RFC3339, rec[4])
		t.EntryPrice, _ = strconv.ParseFloat(rec[5], 64)
		t.ExitPrice, _ = strconv.ParseFloat(rec[6], 64)
		t.Quantity, _ = strconv.ParseFloat(rec[7], 64)
		t.PnL, _ = strconv.ParseFloat(rec[8], 64)
		t.ReturnPct, _ = strconv.ParseFloat(rec[9], 64)
		t.Commission, _ = strconv.ParseFloat(rec[10], 64)
		t.Slippage, _ = strconv.ParseFloat(rec[11], 64)
		tj.trades = append(tj.trades, t)
	}
	return nil
}
