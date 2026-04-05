package storage

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"sync"
	"time"

	"srfm/market-data/aggregator"
)

const (
	alpacaBarsBaseURL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
	historicalDays    = 30
	reqPerSecond      = 10
)

// alpacaBarResponse is the REST API response shape.
type alpacaBarResponse struct {
	Bars          map[string][]alpacaBarItem `json:"bars"`
	NextPageToken *string                    `json:"next_page_token"`
}

type alpacaBarItem struct {
	T string  `json:"t"`
	O float64 `json:"o"`
	H float64 `json:"h"`
	L float64 `json:"l"`
	C float64 `json:"c"`
	V float64 `json:"v"`
}

// HistoricalLoader loads historical bars from Alpaca REST API on startup.
type HistoricalLoader struct {
	store     *BarStore
	cache     *BarCache
	symbols   []string
	apiKey    string
	apiSecret string
	client    *http.Client
}

// NewHistoricalLoader creates a HistoricalLoader.
func NewHistoricalLoader(store *BarStore, cache *BarCache, symbols []string, apiKey, apiSecret string) *HistoricalLoader {
	return &HistoricalLoader{
		store:     store,
		cache:     cache,
		symbols:   symbols,
		apiKey:    apiKey,
		apiSecret: apiSecret,
		client:    &http.Client{Timeout: 30 * time.Second},
	}
}

// rateLimiter is a simple token bucket rate limiter.
type rateLimiter struct {
	mu        sync.Mutex
	tokens    float64
	maxTokens float64
	rate      float64 // tokens per second
	lastTime  time.Time
}

func newRateLimiter(ratePerSec float64) *rateLimiter {
	return &rateLimiter{
		tokens:    ratePerSec,
		maxTokens: ratePerSec,
		rate:      ratePerSec,
		lastTime:  time.Now(),
	}
}

func (r *rateLimiter) Wait() {
	for {
		r.mu.Lock()
		now := time.Now()
		elapsed := now.Sub(r.lastTime).Seconds()
		r.tokens += elapsed * r.rate
		if r.tokens > r.maxTokens {
			r.tokens = r.maxTokens
		}
		r.lastTime = now
		if r.tokens >= 1 {
			r.tokens--
			r.mu.Unlock()
			return
		}
		r.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}
}

// Load fetches historical data for all symbols/timeframes concurrently.
func (l *HistoricalLoader) Load(ctx interface{ Done() <-chan struct{} }) error {
	limiter := newRateLimiter(reqPerSecond)

	end := time.Now().UTC().Truncate(time.Minute)
	start := end.Add(-historicalDays * 24 * time.Hour)

	// Only load 1m bars from API; higher timeframes come from aggregation
	// We also load 1h and 1d directly for faster startup
	apiFetched := []string{"1Min", "1Hour", "1Day"}
	tfMap := map[string]string{"1Min": "1m", "1Hour": "1h", "1Day": "1d"}

	type job struct {
		symbol    string
		alpacaTF  string
		localTF   string
	}

	jobs := make([]job, 0, len(l.symbols)*len(apiFetched))
	for _, sym := range l.symbols {
		for _, atf := range apiFetched {
			jobs = append(jobs, job{sym, atf, tfMap[atf]})
		}
	}

	type result struct {
		symbol  string
		tf      string
		loaded  int
		err     error
	}

	results := make(chan result, len(jobs))
	sem := make(chan struct{}, 5) // 5 concurrent goroutines

	var wg sync.WaitGroup
	for _, j := range jobs {
		select {
		case <-ctx.Done():
			break
		default:
		}
		wg.Add(1)
		j := j
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			limiter.Wait()

			n, err := l.loadSymbol(j.symbol, j.alpacaTF, j.localTF, start, end)
			results <- result{j.symbol, j.localTF, n, err}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	totalLoaded := 0
	errors := 0
	for r := range results {
		if r.err != nil {
			log.Printf("[hist] %s/%s: error: %v", r.symbol, r.tf, r.err)
			errors++
		} else {
			log.Printf("[hist] %s/%s: loaded %d bars", r.symbol, r.tf, r.loaded)
			totalLoaded += r.loaded
		}
	}

	log.Printf("[hist] complete: %d bars loaded, %d errors", totalLoaded, errors)
	return nil
}

func (l *HistoricalLoader) loadSymbol(symbol, alpacaTF, localTF string, start, end time.Time) (int, error) {
	alpacaSym := symbol + "/USD"
	var allBars []aggregator.BarEvent
	var pageToken *string

	for {
		bars, nextToken, err := l.fetchPage(alpacaSym, alpacaTF, start, end, pageToken)
		if err != nil {
			return len(allBars), err
		}
		for _, b := range bars {
			evt := aggregator.BarEvent{
				Symbol:     symbol,
				Timeframe:  localTF,
				Open:       b.O,
				High:       b.H,
				Low:        b.L,
				Close:      b.C,
				Volume:     b.V,
				IsComplete: true,
				Source:     "alpaca_historical",
			}
			ts, err := time.Parse(time.RFC3339Nano, b.T)
			if err == nil {
				evt.Timestamp = ts.UTC()
			}
			allBars = append(allBars, evt)
		}

		if nextToken == nil || len(bars) == 0 {
			break
		}
		pageToken = nextToken
	}

	// Store to SQLite in batches
	const batchSize = 500
	for i := 0; i < len(allBars); i += batchSize {
		end := i + batchSize
		if end > len(allBars) {
			end = len(allBars)
		}
		if err := l.store.InsertBatch(allBars[i:end]); err != nil {
			return len(allBars), fmt.Errorf("batch insert: %w", err)
		}
	}

	// Warm cache with most recent 500
	if len(allBars) > 0 {
		warmBars := allBars
		if len(warmBars) > 500 {
			warmBars = allBars[len(allBars)-500:]
		}
		l.cache.WarmFrom(warmBars)
	}

	return len(allBars), nil
}

func (l *HistoricalLoader) fetchPage(symbol, timeframe string, start, end time.Time, pageToken *string) ([]alpacaBarItem, *string, error) {
	url := fmt.Sprintf("%s?symbols=%s&timeframe=%s&start=%s&end=%s&limit=1000&sort=asc",
		alpacaBarsBaseURL,
		symbol,
		timeframe,
		start.Format(time.RFC3339),
		end.Format(time.RFC3339),
	)
	if pageToken != nil {
		url += "&page_token=" + *pageToken
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("APCA-API-KEY-ID", l.apiKey)
	req.Header.Set("APCA-API-SECRET-KEY", l.apiSecret)

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, nil, fmt.Errorf("status %d: %s", resp.StatusCode, string(body))
	}

	var parsed alpacaBarResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return nil, nil, fmt.Errorf("decode: %w", err)
	}

	bars := parsed.Bars[symbol]
	return bars, parsed.NextPageToken, nil
}
