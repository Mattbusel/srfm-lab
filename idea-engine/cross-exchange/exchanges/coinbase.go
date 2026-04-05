package exchanges

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"time"

	"srfm-lab/idea-engine/cross-exchange/aggregator"
)

const (
	coinbaseTickerBase   = "https://api.coinbase.com/api/v3/brokerage/market/products/%s/ticker"
	coinbasePollInterval = 15 * time.Second
)

// coinbaseProductMap maps Coinbase product IDs to IAE symbols.
var coinbaseProductMap = map[string]string{
	"BTC-USD":  "BTC",
	"ETH-USD":  "ETH",
	"SOL-USD":  "SOL",
	"XRP-USD":  "XRP",
	"ADA-USD":  "ADA",
	"DOGE-USD": "DOGE",
	"AVAX-USD": "AVAX",
	"LINK-USD": "LINK",
	"DOT-USD":  "DOT",
}

// CoinbasePoller polls Coinbase Advanced Trade prices.
type CoinbasePoller struct {
	agg    *aggregator.PriceAggregator
	client *http.Client
	log    *slog.Logger
}

// NewCoinbasePoller creates a new Coinbase poller.
func NewCoinbasePoller(agg *aggregator.PriceAggregator, log *slog.Logger) *CoinbasePoller {
	return &CoinbasePoller{
		agg:    agg,
		client: &http.Client{Timeout: 10 * time.Second},
		log:    log,
	}
}

// Run starts the polling loop, blocking until ctx is cancelled.
func (c *CoinbasePoller) Run(ctx context.Context) {
	ticker := time.NewTicker(coinbasePollInterval)
	defer ticker.Stop()
	c.poll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.poll(ctx)
		}
	}
}

type coinbaseTickerResponse struct {
	BestBid     string `json:"best_bid"`
	BestAsk     string `json:"best_ask"`
	Price       string `json:"price"`
	Volume      string `json:"volume"`
}

func (c *CoinbasePoller) poll(ctx context.Context) {
	for productID, iaeSymbol := range coinbaseProductMap {
		if err := c.fetchProduct(ctx, productID, iaeSymbol); err != nil {
			c.log.Warn("coinbase: fetch product failed", "product", productID, "err", err)
		}
	}
}

func (c *CoinbasePoller) fetchProduct(ctx context.Context, productID, iaeSymbol string) error {
	url := fmt.Sprintf(coinbaseTickerBase, productID)
	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(math.Pow(2, float64(attempt))) * 500 * time.Millisecond
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		req.Header.Set("Accept", "application/json")

		resp, err := c.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode == http.StatusTooManyRequests {
			resp.Body.Close()
			lastErr = fmt.Errorf("rate limited")
			continue
		}

		if resp.StatusCode != http.StatusOK {
			resp.Body.Close()
			return fmt.Errorf("coinbase: %s returned %d", productID, resp.StatusCode)
		}

		var body struct {
			Trades []struct {
				Price string `json:"price"`
			} `json:"trades"`
			BestBid string `json:"best_bid"`
			BestAsk string `json:"best_ask"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			resp.Body.Close()
			return fmt.Errorf("coinbase: decode %s: %w", productID, err)
		}
		resp.Body.Close()

		bid := parseFloat(body.BestBid)
		ask := parseFloat(body.BestAsk)
		if bid <= 0 || ask <= 0 {
			return nil
		}
		mid := (bid + ask) / 2.0

		c.agg.Update(aggregator.ExchangePrice{
			Exchange:  "coinbase",
			Symbol:    iaeSymbol,
			Bid:       bid,
			Ask:       ask,
			Mid:       mid,
			Timestamp: time.Now().UTC(),
		})
		return nil
	}
	return lastErr
}
