// Package exchanges contains per-exchange price pollers.
package exchanges

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"srfm-lab/idea-engine/cross-exchange/aggregator"
)

const (
	binanceSpotURL    = "https://api.binance.com/api/v3/ticker/price"
	binanceFuturesURL = "https://fapi.binance.com/fapi/v1/ticker/price"
	binanceFundingURL = "https://fapi.binance.com/fapi/v1/fundingRate"
	binanceBookURL    = "https://api.binance.com/api/v3/bookTicker"
	binancePollInterval = 10 * time.Second
)

// binanceSymbolMap maps Binance trading pair suffixes to IAE symbols.
var binanceSymbolMap = map[string]string{
	"BTCUSDT":  "BTC",
	"ETHUSDT":  "ETH",
	"SOLUSDT":  "SOL",
	"BNBUSDT":  "BNB",
	"XRPUSDT":  "XRP",
	"ADAUSDT":  "ADA",
	"DOGEUSDT": "DOGE",
	"AVAXUSDT": "AVAX",
	"LINKUSDT": "LINK",
	"DOTUSDT":  "DOT",
}

// BinanceFuturesPrice holds a price from the futures exchange.
type BinanceFuturesPrice struct {
	Symbol string
	Price  float64
	Time   time.Time
}

// BinanceFundingRate holds a funding rate observation.
type BinanceFundingRate struct {
	Symbol      string
	FundingRate float64
	FundingTime time.Time
}

// BinancePoller polls Binance spot and futures prices.
type BinancePoller struct {
	agg          *aggregator.PriceAggregator
	client       *http.Client
	log          *slog.Logger
	futuresMu    interface{} // stored via FuturesPrices()
	futuresPrices map[string]BinanceFuturesPrice
	fundingRates  map[string]BinanceFundingRate
}

// NewBinancePoller creates a new Binance poller.
func NewBinancePoller(agg *aggregator.PriceAggregator, log *slog.Logger) *BinancePoller {
	return &BinancePoller{
		agg:           agg,
		client:        &http.Client{Timeout: 10 * time.Second},
		log:           log,
		futuresPrices: make(map[string]BinanceFuturesPrice),
		fundingRates:  make(map[string]BinanceFundingRate),
	}
}

// Run starts the polling loop, blocking until ctx is cancelled.
func (b *BinancePoller) Run(ctx context.Context) {
	ticker := time.NewTicker(binancePollInterval)
	defer ticker.Stop()
	// Poll immediately on start.
	b.poll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			b.poll(ctx)
		}
	}
}

func (b *BinancePoller) poll(ctx context.Context) {
	b.pollSpot(ctx)
	b.pollFutures(ctx)
	b.pollFunding(ctx)
}

type binanceTickerResponse struct {
	Symbol string `json:"symbol"`
	Price  string `json:"price"`
}

type binanceBookTickerResponse struct {
	Symbol   string `json:"symbol"`
	BidPrice string `json:"bidPrice"`
	BidQty   string `json:"bidQty"`
	AskPrice string `json:"askPrice"`
	AskQty   string `json:"askQty"`
}

func (b *BinancePoller) pollSpot(ctx context.Context) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, binanceBookURL, nil)
	if err != nil {
		b.log.Error("binance: build book ticker request", "err", err)
		return
	}
	resp, err := b.client.Do(req)
	if err != nil {
		b.log.Error("binance: book ticker request failed", "err", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b.log.Error("binance: book ticker non-200", "status", resp.StatusCode)
		return
	}

	var tickers []binanceBookTickerResponse
	if err := json.NewDecoder(resp.Body).Decode(&tickers); err != nil {
		b.log.Error("binance: decode book ticker", "err", err)
		return
	}

	now := time.Now().UTC()
	for _, t := range tickers {
		iaeSymbol, ok := binanceSymbolMap[t.Symbol]
		if !ok {
			continue
		}
		bid := parseFloat(t.BidPrice)
		ask := parseFloat(t.AskPrice)
		if bid <= 0 || ask <= 0 {
			continue
		}
		mid := (bid + ask) / 2.0
		b.agg.Update(aggregator.ExchangePrice{
			Exchange:  "binance",
			Symbol:    iaeSymbol,
			Bid:       bid,
			Ask:       ask,
			Mid:       mid,
			Timestamp: now,
		})
	}
	b.log.Debug("binance: spot prices updated", "count", len(tickers))
}

func (b *BinancePoller) pollFutures(ctx context.Context) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, binanceFuturesURL, nil)
	if err != nil {
		b.log.Error("binance: build futures request", "err", err)
		return
	}
	resp, err := b.client.Do(req)
	if err != nil {
		b.log.Error("binance: futures request failed", "err", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b.log.Error("binance: futures non-200", "status", resp.StatusCode)
		return
	}

	var tickers []binanceTickerResponse
	if err := json.NewDecoder(resp.Body).Decode(&tickers); err != nil {
		b.log.Error("binance: decode futures", "err", err)
		return
	}

	now := time.Now().UTC()
	for _, t := range tickers {
		iaeSymbol, ok := binanceSymbolMap[t.Symbol]
		if !ok {
			continue
		}
		price := parseFloat(t.Price)
		if price <= 0 {
			continue
		}
		b.futuresPrices[iaeSymbol] = BinanceFuturesPrice{
			Symbol: iaeSymbol,
			Price:  price,
			Time:   now,
		}
	}
	b.log.Debug("binance: futures prices updated")
}

type binanceFundingResponse struct {
	Symbol      string `json:"symbol"`
	FundingRate string `json:"fundingRate"`
	FundingTime int64  `json:"fundingTime"`
}

func (b *BinancePoller) pollFunding(ctx context.Context) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, binanceFundingURL, nil)
	if err != nil {
		b.log.Error("binance: build funding request", "err", err)
		return
	}
	resp, err := b.client.Do(req)
	if err != nil {
		b.log.Error("binance: funding request failed", "err", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b.log.Error("binance: funding non-200", "status", resp.StatusCode)
		return
	}

	var rates []binanceFundingResponse
	if err := json.NewDecoder(resp.Body).Decode(&rates); err != nil {
		b.log.Error("binance: decode funding", "err", err)
		return
	}

	for _, r := range rates {
		iaeSymbol, ok := binanceSymbolMap[r.Symbol]
		if !ok {
			// Try stripping the suffix.
			iaeSymbol = mapBinanceFuturesSymbol(r.Symbol)
			if iaeSymbol == "" {
				continue
			}
		}
		b.fundingRates[iaeSymbol] = BinanceFundingRate{
			Symbol:      iaeSymbol,
			FundingRate: parseFloat(r.FundingRate),
			FundingTime: time.UnixMilli(r.FundingTime).UTC(),
		}
	}
}

// FuturesPrices returns a snapshot of the latest futures prices.
func (b *BinancePoller) FuturesPrices() map[string]BinanceFuturesPrice {
	out := make(map[string]BinanceFuturesPrice, len(b.futuresPrices))
	for k, v := range b.futuresPrices {
		out[k] = v
	}
	return out
}

// FundingRates returns a snapshot of the latest funding rates.
func (b *BinancePoller) FundingRates() map[string]BinanceFundingRate {
	out := make(map[string]BinanceFundingRate, len(b.fundingRates))
	for k, v := range b.fundingRates {
		out[k] = v
	}
	return out
}

// mapBinanceFuturesSymbol attempts to strip common suffixes from a perp
// futures symbol (e.g. BTCUSDT → BTC).
func mapBinanceFuturesSymbol(s string) string {
	for _, suffix := range []string{"USDT", "USD", "BUSD"} {
		if strings.HasSuffix(s, suffix) {
			base := strings.TrimSuffix(s, suffix)
			if len(base) >= 2 {
				return base
			}
		}
	}
	return ""
}

// parseFloat converts a string to float64, returning 0 on error.
func parseFloat(s string) float64 {
	if s == "" {
		return 0
	}
	var f float64
	fmt.Sscanf(s, "%f", &f)
	return f
}
