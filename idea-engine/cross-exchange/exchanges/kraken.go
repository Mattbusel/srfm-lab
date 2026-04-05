package exchanges

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"srfm-lab/idea-engine/cross-exchange/aggregator"
)

const (
	krakenTickerURL      = "https://api.kraken.com/0/public/Ticker"
	krakenPollInterval   = 15 * time.Second
)

// krakenPairMap maps Kraken pair names to IAE symbols.
// Kraken uses XBT for Bitcoin and XETHz for Ethereum with various bases.
var krakenPairMap = map[string]string{
	"XBTUSD":  "BTC",
	"XXBTZUSD": "BTC",
	"XETHZUSD": "ETH",
	"ETHUSD":  "ETH",
	"SOLUSD":  "SOL",
	"XRPUSD":  "XRP",
	"XXRPZUSD": "XRP",
	"ADAUSD":  "ADA",
	"XADAZUSD": "ADA",
	"DOGUSD":  "DOGE",
	"AVAXUSD": "AVAX",
	"LINKUSD": "LINK",
	"DOTUSD":  "DOT",
}

// krakenQueryPairs is the comma-separated pair list sent to the API.
const krakenQueryPairs = "XBTUSD,XETHZUSD,SOLUSD,XRPUSD,ADAUSD,DOGUSD,AVAXUSD,LINKUSD,DOTUSD"

// KrakenPoller polls Kraken public ticker prices.
type KrakenPoller struct {
	agg    *aggregator.PriceAggregator
	client *http.Client
	log    *slog.Logger
}

// NewKrakenPoller creates a new Kraken poller.
func NewKrakenPoller(agg *aggregator.PriceAggregator, log *slog.Logger) *KrakenPoller {
	return &KrakenPoller{
		agg:    agg,
		client: &http.Client{Timeout: 10 * time.Second},
		log:    log,
	}
}

// Run starts the polling loop, blocking until ctx is cancelled.
func (k *KrakenPoller) Run(ctx context.Context) {
	ticker := time.NewTicker(krakenPollInterval)
	defer ticker.Stop()
	k.poll(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			k.poll(ctx)
		}
	}
}

// krakenTickerData represents Kraken's quirky per-pair ticker object.
// Arrays are [today, last_24h] for most fields.
// b = best bid [price, whole_lot_vol, lot_vol]
// a = best ask [price, whole_lot_vol, lot_vol]
// c = last trade [price, lot_vol]
// v = volume [today, last_24h]
type krakenTickerData struct {
	Bid    [3]json.RawMessage `json:"b"` // [price, whole_lot, lot]
	Ask    [3]json.RawMessage `json:"a"`
	Last   [2]json.RawMessage `json:"c"` // [price, lot_vol]
	Volume [2]json.RawMessage `json:"v"` // [today, last_24h]
}

type krakenTickerResponse struct {
	Error  []string                     `json:"error"`
	Result map[string]krakenTickerData  `json:"result"`
}

func (k *KrakenPoller) poll(ctx context.Context) {
	url := fmt.Sprintf("%s?pair=%s", krakenTickerURL, krakenQueryPairs)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		k.log.Error("kraken: build request", "err", err)
		return
	}

	resp, err := k.client.Do(req)
	if err != nil {
		k.log.Error("kraken: request failed", "err", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		k.log.Error("kraken: non-200", "status", resp.StatusCode)
		return
	}

	var body krakenTickerResponse
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		k.log.Error("kraken: decode response", "err", err)
		return
	}

	if len(body.Error) > 0 {
		k.log.Warn("kraken: api errors", "errors", body.Error)
	}

	now := time.Now().UTC()
	for pairName, data := range body.Result {
		iaeSymbol := krakenPairMap[pairName]
		if iaeSymbol == "" {
			// Try case-insensitive lookup or strip trailing digits.
			iaeSymbol = krakenPairMap[stripKrakenSuffix(pairName)]
		}
		if iaeSymbol == "" {
			continue
		}

		bid := parseKrakenPrice(data.Bid[0])
		ask := parseKrakenPrice(data.Ask[0])
		vol := parseKrakenVolume(data.Volume[1]) // 24h volume
		if bid <= 0 || ask <= 0 {
			continue
		}
		mid := (bid + ask) / 2.0

		k.agg.Update(aggregator.ExchangePrice{
			Exchange:  "kraken",
			Symbol:    iaeSymbol,
			Bid:       bid,
			Ask:       ask,
			Mid:       mid,
			Volume:    vol,
			Timestamp: now,
		})
	}
	k.log.Debug("kraken: prices updated", "pairs", len(body.Result))
}

// parseKrakenPrice decodes a JSON-encoded price string from a Kraken array element.
func parseKrakenPrice(raw json.RawMessage) float64 {
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return 0
	}
	return parseFloat(s)
}

// parseKrakenVolume decodes a JSON-encoded volume string.
func parseKrakenVolume(raw json.RawMessage) float64 {
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return 0
	}
	return parseFloat(s)
}

// stripKrakenSuffix removes known quote currency suffixes.
func stripKrakenSuffix(pair string) string {
	for _, suffix := range []string{"USD", "ZUSD", "EUR", "ZEUR"} {
		if len(pair) > len(suffix) && pair[len(pair)-len(suffix):] == suffix {
			return pair[:len(pair)-len(suffix)]
		}
	}
	return pair
}
