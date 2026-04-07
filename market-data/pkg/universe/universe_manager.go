// Package universe manages the set of tradeable instruments (the "universe")
// and provides filtering and lookup capabilities.
package universe

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// InstrumentInfo describes a single tradeable instrument.
type InstrumentInfo struct {
	Symbol     string  `json:"symbol"`
	Name       string  `json:"name"`
	Exchange   string  `json:"exchange"`
	AssetClass string  `json:"asset_class"` // "equity", "crypto", "etf", etc.
	Sector     string  `json:"sector"`
	MarketCap  float64 `json:"market_cap_usd"`
	ADV30      float64 `json:"adv30_usd"` // 30-day average daily volume in USD
	IsActive   bool    `json:"is_active"`
}

// DefaultUniverse is the built-in set of instruments used when no file is loaded.
// It covers US large-cap equities and top crypto assets.
var DefaultUniverse = []InstrumentInfo{
	{Symbol: "AAPL", Name: "Apple Inc.", Exchange: "NASDAQ", AssetClass: "equity", Sector: "technology", MarketCap: 2_900_000_000_000, ADV30: 8_000_000_000, IsActive: true},
	{Symbol: "MSFT", Name: "Microsoft Corp.", Exchange: "NASDAQ", AssetClass: "equity", Sector: "technology", MarketCap: 2_700_000_000_000, ADV30: 5_000_000_000, IsActive: true},
	{Symbol: "GOOGL", Name: "Alphabet Inc.", Exchange: "NASDAQ", AssetClass: "equity", Sector: "technology", MarketCap: 2_100_000_000_000, ADV30: 3_000_000_000, IsActive: true},
	{Symbol: "AMZN", Name: "Amazon.com Inc.", Exchange: "NASDAQ", AssetClass: "equity", Sector: "consumer_discretionary", MarketCap: 1_900_000_000_000, ADV30: 4_500_000_000, IsActive: true},
	{Symbol: "META", Name: "Meta Platforms Inc.", Exchange: "NASDAQ", AssetClass: "equity", Sector: "communication_services", MarketCap: 1_200_000_000_000, ADV30: 2_500_000_000, IsActive: true},
	{Symbol: "TSLA", Name: "Tesla Inc.", Exchange: "NASDAQ", AssetClass: "equity", Sector: "consumer_discretionary", MarketCap: 700_000_000_000, ADV30: 15_000_000_000, IsActive: true},
	{Symbol: "BTC-USD", Name: "Bitcoin", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "layer1", MarketCap: 1_300_000_000_000, ADV30: 20_000_000_000, IsActive: true},
	{Symbol: "ETH-USD", Name: "Ethereum", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "layer1", MarketCap: 400_000_000_000, ADV30: 8_000_000_000, IsActive: true},
	{Symbol: "SOL-USD", Name: "Solana", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "layer1", MarketCap: 90_000_000_000, ADV30: 2_000_000_000, IsActive: true},
	{Symbol: "BNB-USD", Name: "BNB", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "exchange_token", MarketCap: 80_000_000_000, ADV30: 1_200_000_000, IsActive: true},
	{Symbol: "ARB-USD", Name: "Arbitrum", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "layer2", MarketCap: 3_000_000_000, ADV30: 150_000_000, IsActive: true},
	{Symbol: "AVAX-USD", Name: "Avalanche", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "layer1", MarketCap: 12_000_000_000, ADV30: 400_000_000, IsActive: true},
	{Symbol: "LINK-USD", Name: "Chainlink", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "oracle", MarketCap: 10_000_000_000, ADV30: 350_000_000, IsActive: true},
	{Symbol: "OP-USD", Name: "Optimism", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "layer2", MarketCap: 2_500_000_000, ADV30: 120_000_000, IsActive: true},
	{Symbol: "MATIC-USD", Name: "Polygon", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "layer2", MarketCap: 5_000_000_000, ADV30: 300_000_000, IsActive: true},
	{Symbol: "NVDA", Name: "NVIDIA Corp.", Exchange: "NASDAQ", AssetClass: "equity", Sector: "technology", MarketCap: 2_200_000_000_000, ADV30: 10_000_000_000, IsActive: true},
	{Symbol: "SPY", Name: "SPDR S&P 500 ETF", Exchange: "NYSE", AssetClass: "etf", Sector: "broad_market", MarketCap: 500_000_000_000, ADV30: 25_000_000_000, IsActive: true},
	{Symbol: "QQQ", Name: "Invesco QQQ ETF", Exchange: "NASDAQ", AssetClass: "etf", Sector: "technology", MarketCap: 200_000_000_000, ADV30: 12_000_000_000, IsActive: true},
	{Symbol: "JPM", Name: "JPMorgan Chase", Exchange: "NYSE", AssetClass: "equity", Sector: "financials", MarketCap: 550_000_000_000, ADV30: 2_200_000_000, IsActive: true},
	{Symbol: "GS", Name: "Goldman Sachs", Exchange: "NYSE", AssetClass: "equity", Sector: "financials", MarketCap: 150_000_000_000, ADV30: 700_000_000, IsActive: true},
	{Symbol: "XRP-USD", Name: "XRP", Exchange: "CRYPTO", AssetClass: "crypto", Sector: "payments", MarketCap: 30_000_000_000, ADV30: 1_500_000_000, IsActive: true},
}

// UniverseManager holds the active instrument universe and provides fast
// lookup and filtering.  All methods are safe for concurrent use.
type UniverseManager struct {
	mu          sync.RWMutex
	instruments []InstrumentInfo
	bySymbol    map[string]InstrumentInfo
}

// NewUniverseManager returns a manager pre-loaded with the DefaultUniverse.
func NewUniverseManager() *UniverseManager {
	m := &UniverseManager{}
	m.loadSlice(DefaultUniverse)
	return m
}

// LoadUniverse replaces the current universe by reading from path.
// Supported formats: .json (array of InstrumentInfo) and .csv.
func (m *UniverseManager) LoadUniverse(path string) error {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".json":
		return m.loadJSON(path)
	case ".csv":
		return m.loadCSV(path)
	default:
		return fmt.Errorf("unsupported file format: %q (want .json or .csv)", ext)
	}
}

// ActiveSymbols returns all symbols where IsActive == true, sorted alphabetically.
func (m *UniverseManager) ActiveSymbols() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var out []string
	for _, inst := range m.instruments {
		if inst.IsActive {
			out = append(out, inst.Symbol)
		}
	}
	sortStrings(out)
	return out
}

// AllSymbols returns every symbol in the universe regardless of active status.
func (m *UniverseManager) AllSymbols() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make([]string, len(m.instruments))
	for i, inst := range m.instruments {
		out[i] = inst.Symbol
	}
	return out
}

// SymbolInfo returns the InstrumentInfo for the given symbol (case-insensitive).
// Returns false if the symbol is not in the universe.
func (m *UniverseManager) SymbolInfo(symbol string) (InstrumentInfo, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	info, ok := m.bySymbol[strings.ToUpper(symbol)]
	return info, ok
}

// FilterByADV returns all active symbols with ADV30 >= minADV (in USD).
func (m *UniverseManager) FilterByADV(minADV float64) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var out []string
	for _, inst := range m.instruments {
		if inst.IsActive && inst.ADV30 >= minADV {
			out = append(out, inst.Symbol)
		}
	}
	sortStrings(out)
	return out
}

// FilterBySector returns all active symbols in the given sector (case-insensitive).
func (m *UniverseManager) FilterBySector(sector string) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	target := strings.ToLower(strings.TrimSpace(sector))
	var out []string
	for _, inst := range m.instruments {
		if inst.IsActive && strings.ToLower(inst.Sector) == target {
			out = append(out, inst.Symbol)
		}
	}
	sortStrings(out)
	return out
}

// FilterByAssetClass returns active symbols matching the given asset class.
func (m *UniverseManager) FilterByAssetClass(assetClass string) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	target := strings.ToLower(strings.TrimSpace(assetClass))
	var out []string
	for _, inst := range m.instruments {
		if inst.IsActive && strings.ToLower(inst.AssetClass) == target {
			out = append(out, inst.Symbol)
		}
	}
	sortStrings(out)
	return out
}

// SetActive enables or disables a symbol without reloading the full universe.
func (m *UniverseManager) SetActive(symbol string, active bool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	sym := strings.ToUpper(symbol)
	info, ok := m.bySymbol[sym]
	if !ok {
		return fmt.Errorf("symbol %q not in universe", symbol)
	}
	info.IsActive = active
	m.bySymbol[sym] = info

	for i := range m.instruments {
		if m.instruments[i].Symbol == sym {
			m.instruments[i].IsActive = active
			break
		}
	}
	return nil
}

// Size returns the total number of instruments in the universe.
func (m *UniverseManager) Size() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.instruments)
}

// -- internal helpers --

func (m *UniverseManager) loadSlice(instruments []InstrumentInfo) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.instruments = make([]InstrumentInfo, len(instruments))
	copy(m.instruments, instruments)
	m.bySymbol = make(map[string]InstrumentInfo, len(instruments))
	for _, inst := range instruments {
		m.bySymbol[strings.ToUpper(inst.Symbol)] = inst
	}
}

func (m *UniverseManager) loadJSON(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var instruments []InstrumentInfo
	if err := json.NewDecoder(f).Decode(&instruments); err != nil {
		return fmt.Errorf("decode json: %w", err)
	}
	m.loadSlice(instruments)
	return nil
}

func (m *UniverseManager) loadCSV(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		return fmt.Errorf("read csv: %w", err)
	}
	if len(records) < 2 {
		return fmt.Errorf("csv has no data rows")
	}

	// Expect header: symbol,name,exchange,asset_class,sector,market_cap,adv30,is_active
	var instruments []InstrumentInfo
	for _, row := range records[1:] {
		if len(row) < 8 {
			continue
		}
		mc, _ := strconv.ParseFloat(row[5], 64)
		adv, _ := strconv.ParseFloat(row[6], 64)
		active := strings.EqualFold(strings.TrimSpace(row[7]), "true")
		instruments = append(instruments, InstrumentInfo{
			Symbol:     strings.ToUpper(strings.TrimSpace(row[0])),
			Name:       row[1],
			Exchange:   row[2],
			AssetClass: row[3],
			Sector:     row[4],
			MarketCap:  mc,
			ADV30:      adv,
			IsActive:   active,
		})
	}
	m.loadSlice(instruments)
	return nil
}

// sortStrings sorts a string slice in place using a simple insertion sort.
// For small universe slices this avoids a sort package import.
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		key := s[i]
		j := i - 1
		for j >= 0 && s[j] > key {
			s[j+1] = s[j]
			j--
		}
		s[j+1] = key
	}
}
