// Package client provides a typed gRPC client for all SRFM microservices.
// Features: connection pooling, automatic retry with exponential backoff,
// circuit breaker (gobreaker), and TLS/mTLS support.
package client

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sony/gobreaker"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"

	pbm  "github.com/srfm/infra/grpc/proto/market"
	pbp  "github.com/srfm/infra/grpc/proto/portfolio"
	pbr  "github.com/srfm/infra/grpc/proto/risk"
	pbs  "github.com/srfm/infra/grpc/proto/strategy"
)

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

// ClientConfig holds connection settings for a single service endpoint.
type ClientConfig struct {
	// Target is the gRPC address, e.g. "localhost:50051" or a DNS address.
	Target string

	// PoolSize controls how many underlying connections are maintained.
	PoolSize int

	// TLS configuration. If CACert is empty, insecure mode is used.
	CACert     string // path to CA certificate PEM
	ClientCert string // path to client certificate PEM (mTLS)
	ClientKey  string // path to client key PEM (mTLS)

	// Retry settings.
	MaxRetries       int
	InitialBackoff   time.Duration
	MaxBackoff       time.Duration
	BackoffMultiplier float64

	// Circuit breaker.
	CBMaxRequests   uint32        // requests allowed in half-open state
	CBInterval      time.Duration // rolling window
	CBTimeout       time.Duration // how long to stay open after tripping
	CBMinRequests   uint32        // min requests before CB engages
	CBFailureRatio  float64       // failure ratio to trip CB

	// Timeouts.
	DialTimeout    time.Duration
	RequestTimeout time.Duration
	KeepaliveTime  time.Duration
	KeepaliveTimeout time.Duration
}

// DefaultClientConfig returns production-ready defaults.
func DefaultClientConfig(target string) ClientConfig {
	return ClientConfig{
		Target:            target,
		PoolSize:          4,
		MaxRetries:        3,
		InitialBackoff:    100 * time.Millisecond,
		MaxBackoff:        5 * time.Second,
		BackoffMultiplier: 2.0,
		CBMaxRequests:     5,
		CBInterval:        60 * time.Second,
		CBTimeout:         30 * time.Second,
		CBMinRequests:     10,
		CBFailureRatio:    0.6,
		DialTimeout:       10 * time.Second,
		RequestTimeout:    30 * time.Second,
		KeepaliveTime:     30 * time.Second,
		KeepaliveTimeout:  10 * time.Second,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection pool
// ─────────────────────────────────────────────────────────────────────────────

type connEntry struct {
	conn    *grpc.ClientConn
	inFlight int64 // atomic
}

type connPool struct {
	conns   []*connEntry
	size    int
	counter uint64 // atomic round-robin counter
}

func newConnPool(cfg ClientConfig) (*connPool, error) {
	dialOpts, err := buildDialOptions(cfg)
	if err != nil {
		return nil, fmt.Errorf("dial options: %w", err)
	}

	pool := &connPool{
		size:  cfg.PoolSize,
		conns: make([]*connEntry, cfg.PoolSize),
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.DialTimeout)
	defer cancel()

	for i := 0; i < cfg.PoolSize; i++ {
		conn, err := grpc.DialContext(ctx, cfg.Target, dialOpts...) //nolint:staticcheck
		if err != nil {
			// Close already-opened connections.
			for j := 0; j < i; j++ {
				_ = pool.conns[j].conn.Close()
			}
			return nil, fmt.Errorf("dial [%d/%d] to %s: %w", i+1, cfg.PoolSize, cfg.Target, err)
		}
		pool.conns[i] = &connEntry{conn: conn}
	}
	return pool, nil
}

// get returns the least-loaded connection (round-robin with tiebreak on inflight).
func (p *connPool) get() *grpc.ClientConn {
	idx := atomic.AddUint64(&p.counter, 1) % uint64(p.size)
	entry := p.conns[idx]
	atomic.AddInt64(&entry.inFlight, 1)
	return entry.conn
}

func (p *connPool) close() {
	for _, e := range p.conns {
		_ = e.conn.Close()
	}
}

// buildDialOptions constructs grpc.DialOptions from config.
func buildDialOptions(cfg ClientConfig) ([]grpc.DialOption, error) {
	var opts []grpc.DialOption

	// TLS / insecure.
	if cfg.CACert == "" {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	} else {
		caCert, err := os.ReadFile(cfg.CACert)
		if err != nil {
			return nil, fmt.Errorf("read CA cert: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to parse CA cert")
		}
		tlsCfg := &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS13}
		if cfg.ClientCert != "" && cfg.ClientKey != "" {
			cert, err := tls.LoadX509KeyPair(cfg.ClientCert, cfg.ClientKey)
			if err != nil {
				return nil, fmt.Errorf("load client cert/key: %w", err)
			}
			tlsCfg.Certificates = []tls.Certificate{cert}
		}
		opts = append(opts, grpc.WithTransportCredentials(credentials.NewTLS(tlsCfg)))
	}

	// Keepalive.
	opts = append(opts, grpc.WithKeepaliveParams(keepalive.ClientParameters{
		Time:                cfg.KeepaliveTime,
		Timeout:             cfg.KeepaliveTimeout,
		PermitWithoutStream: true,
	}))

	// Default call options.
	opts = append(opts, grpc.WithDefaultCallOptions(
		grpc.MaxCallRecvMsgSize(32*1024*1024),
		grpc.MaxCallSendMsgSize(32*1024*1024),
	))

	return opts, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Retry logic
// ─────────────────────────────────────────────────────────────────────────────

// isRetryable returns true for transient gRPC status codes.
func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	s, ok := status.FromError(err)
	if !ok {
		return false
	}
	switch s.Code() {
	case codes.Unavailable, codes.DeadlineExceeded, codes.ResourceExhausted:
		return true
	}
	return false
}

// retryConfig bundles retry parameters.
type retryConfig struct {
	maxRetries        int
	initialBackoff    time.Duration
	maxBackoff        time.Duration
	backoffMultiplier float64
}

// withRetry executes fn with exponential backoff retries.
func withRetry(ctx context.Context, cfg retryConfig, fn func() error) error {
	backoff := cfg.initialBackoff
	var lastErr error
	for attempt := 0; attempt <= cfg.maxRetries; attempt++ {
		lastErr = fn()
		if lastErr == nil {
			return nil
		}
		if !isRetryable(lastErr) {
			return lastErr
		}
		if attempt == cfg.maxRetries {
			break
		}
		// Jittered sleep.
		jitter := time.Duration(float64(backoff) * (0.8 + 0.4*rand.Float64()))
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(jitter):
		}
		backoff = time.Duration(float64(backoff) * cfg.backoffMultiplier)
		if backoff > cfg.maxBackoff {
			backoff = cfg.maxBackoff
		}
	}
	return fmt.Errorf("all %d retries exhausted: %w", cfg.maxRetries, lastErr)
}

// ─────────────────────────────────────────────────────────────────────────────
// Circuit breaker wrapper
// ─────────────────────────────────────────────────────────────────────────────

func newCircuitBreaker(name string, cfg ClientConfig) *gobreaker.CircuitBreaker {
	return gobreaker.NewCircuitBreaker(gobreaker.Settings{
		Name:        name,
		MaxRequests: cfg.CBMaxRequests,
		Interval:    cfg.CBInterval,
		Timeout:     cfg.CBTimeout,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			if counts.Requests < uint32(cfg.CBMinRequests) {
				return false
			}
			failRatio := float64(counts.TotalFailures) / float64(counts.Requests)
			return failRatio >= cfg.CBFailureRatio
		},
		OnStateChange: func(name string, from, to gobreaker.State) {
			// In prod: emit metric / log.
			_ = name
			_ = from
			_ = to
		},
	})
}

// ─────────────────────────────────────────────────────────────────────────────
// MarketDataClient
// ─────────────────────────────────────────────────────────────────────────────

// MarketDataClient wraps pbm.MarketDataServiceClient with pooling, retry, CB.
type MarketDataClient struct {
	pool    *connPool
	cfg     ClientConfig
	retry   retryConfig
	cb      *gobreaker.CircuitBreaker
	log     *zap.Logger
}

// NewMarketDataClient constructs a MarketDataClient and opens the connection pool.
func NewMarketDataClient(cfg ClientConfig, log *zap.Logger) (*MarketDataClient, error) {
	pool, err := newConnPool(cfg)
	if err != nil {
		return nil, err
	}
	return &MarketDataClient{
		pool: pool,
		cfg:  cfg,
		retry: retryConfig{
			maxRetries:        cfg.MaxRetries,
			initialBackoff:    cfg.InitialBackoff,
			maxBackoff:        cfg.MaxBackoff,
			backoffMultiplier: cfg.BackoffMultiplier,
		},
		cb:  newCircuitBreaker("market-data", cfg),
		log: log,
	}, nil
}

// Close releases all connections.
func (c *MarketDataClient) Close() { c.pool.close() }

func (c *MarketDataClient) client() pbm.MarketDataServiceClient {
	return pbm.NewMarketDataServiceClient(c.pool.get())
}

// GetBars fetches historical bars with retry and circuit breaker.
func (c *MarketDataClient) GetBars(ctx context.Context, req *pbm.BarRequest, opts ...grpc.CallOption) (*pbm.BarResponse, error) {
	var resp *pbm.BarResponse
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()

	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().GetBars(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// GetQuote fetches the latest NBBO quote.
func (c *MarketDataClient) GetQuote(ctx context.Context, req *pbm.QuoteRequest, opts ...grpc.CallOption) (*pbm.QuoteResponse, error) {
	var resp *pbm.QuoteResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().GetQuote(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// GetOrderBook fetches the current order book snapshot.
func (c *MarketDataClient) GetOrderBook(ctx context.Context, req *pbm.OrderBookRequest, opts ...grpc.CallOption) (*pbm.OrderBookResponse, error) {
	var resp *pbm.OrderBookResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().GetOrderBook(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// StreamBars opens a streaming RPC for real-time bars.
// Returns the stream; the caller is responsible for reading and closing.
func (c *MarketDataClient) StreamBars(ctx context.Context, req *pbm.StreamBarsRequest, opts ...grpc.CallOption) (pbm.MarketDataService_StreamBarsClient, error) {
	return c.client().StreamBars(ctx, req, opts...)
}

// StreamOrderBook opens a streaming order book feed.
func (c *MarketDataClient) StreamOrderBook(ctx context.Context, req *pbm.OrderBookRequest, opts ...grpc.CallOption) (pbm.MarketDataService_StreamOrderBookClient, error) {
	return c.client().StreamOrderBook(ctx, req, opts...)
}

// ─────────────────────────────────────────────────────────────────────────────
// StrategyClient
// ─────────────────────────────────────────────────────────────────────────────

// StrategyClient wraps pbs.StrategyServiceClient.
type StrategyClient struct {
	pool  *connPool
	cfg   ClientConfig
	retry retryConfig
	cb    *gobreaker.CircuitBreaker
	log   *zap.Logger
}

// NewStrategyClient constructs a StrategyClient.
func NewStrategyClient(cfg ClientConfig, log *zap.Logger) (*StrategyClient, error) {
	pool, err := newConnPool(cfg)
	if err != nil {
		return nil, err
	}
	return &StrategyClient{
		pool: pool,
		cfg:  cfg,
		retry: retryConfig{
			maxRetries:        cfg.MaxRetries,
			initialBackoff:    cfg.InitialBackoff,
			maxBackoff:        cfg.MaxBackoff,
			backoffMultiplier: cfg.BackoffMultiplier,
		},
		cb:  newCircuitBreaker("strategy", cfg),
		log: log,
	}, nil
}

func (c *StrategyClient) Close() { c.pool.close() }

func (c *StrategyClient) client() pbs.StrategyServiceClient {
	return pbs.NewStrategyServiceClient(c.pool.get())
}

// ComputeSignal computes a directional signal with retry.
func (c *StrategyClient) ComputeSignal(ctx context.Context, req *pbs.SignalRequest, opts ...grpc.CallOption) (*pbs.SignalResponse, error) {
	var resp *pbs.SignalResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().ComputeSignal(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// GetBHState fetches the full BH state vector.
func (c *StrategyClient) GetBHState(ctx context.Context, req *pbs.BHStateRequest, opts ...grpc.CallOption) (*pbs.BHStateResponse, error) {
	var resp *pbs.BHStateResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().GetBHState(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// GetDeltaScore fetches a delta score for a hypothesis bucket.
func (c *StrategyClient) GetDeltaScore(ctx context.Context, req *pbs.DeltaScoreRequest, opts ...grpc.CallOption) (*pbs.DeltaScoreResponse, error) {
	var resp *pbs.DeltaScoreResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().GetDeltaScore(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// BatchComputeSignals computes signals for multiple symbols concurrently.
func (c *StrategyClient) BatchComputeSignals(ctx context.Context, req *pbs.BatchSignalRequest, opts ...grpc.CallOption) (*pbs.BatchSignalResponse, error) {
	var resp *pbs.BatchSignalResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout*2)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().BatchComputeSignals(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// StreamSignals opens a real-time signal stream.
func (c *StrategyClient) StreamSignals(ctx context.Context, req *pbs.StreamSignalsRequest, opts ...grpc.CallOption) (pbs.StrategyService_StreamSignalsClient, error) {
	return c.client().StreamSignals(ctx, req, opts...)
}

// ─────────────────────────────────────────────────────────────────────────────
// RiskClient
// ─────────────────────────────────────────────────────────────────────────────

// RiskClient wraps pbr.RiskServiceClient.
type RiskClient struct {
	pool  *connPool
	cfg   ClientConfig
	retry retryConfig
	cb    *gobreaker.CircuitBreaker
	log   *zap.Logger
}

// NewRiskClient constructs a RiskClient.
func NewRiskClient(cfg ClientConfig, log *zap.Logger) (*RiskClient, error) {
	pool, err := newConnPool(cfg)
	if err != nil {
		return nil, err
	}
	return &RiskClient{
		pool: pool,
		cfg:  cfg,
		retry: retryConfig{
			maxRetries:        cfg.MaxRetries,
			initialBackoff:    cfg.InitialBackoff,
			maxBackoff:        cfg.MaxBackoff,
			backoffMultiplier: cfg.BackoffMultiplier,
		},
		cb:  newCircuitBreaker("risk", cfg),
		log: log,
	}, nil
}

func (c *RiskClient) Close() { c.pool.close() }

func (c *RiskClient) client() pbr.RiskServiceClient {
	return pbr.NewRiskServiceClient(c.pool.get())
}

// CheckPreTrade performs a pre-trade risk check.
func (c *RiskClient) CheckPreTrade(ctx context.Context, req *pbr.PreTradeRequest, opts ...grpc.CallOption) (*pbr.PreTradeResponse, error) {
	var resp *pbr.PreTradeResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, 5*time.Second) // tight SLA for pre-trade
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, retryConfig{maxRetries: 1, initialBackoff: 50 * time.Millisecond, maxBackoff: 200 * time.Millisecond, backoffMultiplier: 2}, func() error {
			var err error
			resp, err = c.client().CheckPreTrade(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// GetVaR computes Value-at-Risk.
func (c *RiskClient) GetVaR(ctx context.Context, req *pbr.VaRRequest, opts ...grpc.CallOption) (*pbr.VaRResponse, error) {
	var resp *pbr.VaRResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().GetVaR(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// GetPortfolioRisk fetches the full portfolio risk summary.
func (c *RiskClient) GetPortfolioRisk(ctx context.Context, req *pbr.PortfolioRiskRequest, opts ...grpc.CallOption) (*pbr.PortfolioRiskResponse, error) {
	var resp *pbr.PortfolioRiskResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().GetPortfolioRisk(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// StreamRiskEvents opens a real-time risk event stream.
func (c *RiskClient) StreamRiskEvents(ctx context.Context, req *pbr.StreamRiskRequest, opts ...grpc.CallOption) (pbr.RiskService_StreamRiskEventsClient, error) {
	return c.client().StreamRiskEvents(ctx, req, opts...)
}

// ─────────────────────────────────────────────────────────────────────────────
// PortfolioClient
// ─────────────────────────────────────────────────────────────────────────────

// PortfolioClient wraps pbp.PortfolioServiceClient.
type PortfolioClient struct {
	pool  *connPool
	cfg   ClientConfig
	retry retryConfig
	cb    *gobreaker.CircuitBreaker
	log   *zap.Logger
}

// NewPortfolioClient constructs a PortfolioClient.
func NewPortfolioClient(cfg ClientConfig, log *zap.Logger) (*PortfolioClient, error) {
	pool, err := newConnPool(cfg)
	if err != nil {
		return nil, err
	}
	return &PortfolioClient{
		pool: pool,
		cfg:  cfg,
		retry: retryConfig{
			maxRetries:        cfg.MaxRetries,
			initialBackoff:    cfg.InitialBackoff,
			maxBackoff:        cfg.MaxBackoff,
			backoffMultiplier: cfg.BackoffMultiplier,
		},
		cb:  newCircuitBreaker("portfolio", cfg),
		log: log,
	}, nil
}

func (c *PortfolioClient) Close() { c.pool.close() }

func (c *PortfolioClient) client() pbp.PortfolioServiceClient {
	return pbp.NewPortfolioServiceClient(c.pool.get())
}

// GetPositions fetches open positions.
func (c *PortfolioClient) GetPositions(ctx context.Context, req *pbp.PositionsRequest, opts ...grpc.CallOption) (*pbp.PositionResponse, error) {
	var resp *pbp.PositionResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().GetPositions(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// GetPnL fetches P&L summary.
func (c *PortfolioClient) GetPnL(ctx context.Context, req *pbp.PnLRequest, opts ...grpc.CallOption) (*pbp.PnLResponse, error) {
	var resp *pbp.PnLResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().GetPnL(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// OptimizeWeights runs mean-variance optimization.
func (c *PortfolioClient) OptimizeWeights(ctx context.Context, req *pbp.OptimizeRequest, opts ...grpc.CallOption) (*pbp.OptimizeResponse, error) {
	var resp *pbp.OptimizeResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, 60*time.Second) // optimization may take longer
	defer cancel()
	_, err := c.cb.Execute(func() (interface{}, error) {
		return nil, withRetry(timeoutCtx, c.retry, func() error {
			var err error
			resp, err = c.client().OptimizeWeights(timeoutCtx, req, opts...)
			return err
		})
	})
	return resp, err
}

// RecordExecution records a trade fill and updates positions.
func (c *PortfolioClient) RecordExecution(ctx context.Context, req *pbp.ExecutionRequest, opts ...grpc.CallOption) (*pbp.ExecutionResponse, error) {
	var resp *pbp.ExecutionResponse
	timeoutCtx, cancel := context.WithTimeout(ctx, c.cfg.RequestTimeout)
	defer cancel()
	// Execution recording should NOT retry — idempotency is not guaranteed.
	resp, err := c.client().RecordExecution(timeoutCtx, req, opts...)
	return resp, err
}

// StreamPositions opens a real-time position update stream.
func (c *PortfolioClient) StreamPositions(ctx context.Context, req *pbp.PositionsRequest, opts ...grpc.CallOption) (pbp.PortfolioService_StreamPositionsClient, error) {
	return c.client().StreamPositions(ctx, req, opts...)
}

// ─────────────────────────────────────────────────────────────────────────────
// SRFMClient — unified facade over all four services
// ─────────────────────────────────────────────────────────────────────────────

// ServiceAddrs holds endpoint addresses for all four gRPC services.
type ServiceAddrs struct {
	Market    string
	Strategy  string
	Risk      string
	Portfolio string
}

// SRFMClient bundles all four typed clients.
type SRFMClient struct {
	Market    *MarketDataClient
	Strategy  *StrategyClient
	Risk      *RiskClient
	Portfolio *PortfolioClient

	mu     sync.Mutex
	closed bool
}

// NewSRFMClient constructs all four service clients from a ServiceAddrs.
// If an address is empty, that client is nil.
func NewSRFMClient(addrs ServiceAddrs, log *zap.Logger) (*SRFMClient, error) {
	sc := &SRFMClient{}
	var err error

	if addrs.Market != "" {
		sc.Market, err = NewMarketDataClient(DefaultClientConfig(addrs.Market), log)
		if err != nil {
			return nil, fmt.Errorf("market client: %w", err)
		}
	}
	if addrs.Strategy != "" {
		sc.Strategy, err = NewStrategyClient(DefaultClientConfig(addrs.Strategy), log)
		if err != nil {
			sc.Close()
			return nil, fmt.Errorf("strategy client: %w", err)
		}
	}
	if addrs.Risk != "" {
		sc.Risk, err = NewRiskClient(DefaultClientConfig(addrs.Risk), log)
		if err != nil {
			sc.Close()
			return nil, fmt.Errorf("risk client: %w", err)
		}
	}
	if addrs.Portfolio != "" {
		sc.Portfolio, err = NewPortfolioClient(DefaultClientConfig(addrs.Portfolio), log)
		if err != nil {
			sc.Close()
			return nil, fmt.Errorf("portfolio client: %w", err)
		}
	}
	return sc, nil
}

// Close releases all service client connections.
func (sc *SRFMClient) Close() {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	if sc.closed {
		return
	}
	sc.closed = true
	if sc.Market != nil {
		sc.Market.Close()
	}
	if sc.Strategy != nil {
		sc.Strategy.Close()
	}
	if sc.Risk != nil {
		sc.Risk.Close()
	}
	if sc.Portfolio != nil {
		sc.Portfolio.Close()
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Health check
// ─────────────────────────────────────────────────────────────────────────────

// HealthStatus represents the state of each service connection.
type HealthStatus struct {
	Market    string
	Strategy  string
	Risk      string
	Portfolio string
}

// Health performs lightweight health probes and returns per-service status.
func (sc *SRFMClient) Health(ctx context.Context) HealthStatus {
	probe := func(name string, fn func() error) string {
		ctx2, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		_ = ctx2
		err := fn()
		if err == nil {
			return "ok"
		}
		return fmt.Sprintf("error: %v", err)
	}

	hs := HealthStatus{
		Market:    "not configured",
		Strategy:  "not configured",
		Risk:      "not configured",
		Portfolio: "not configured",
	}

	if sc.Market != nil {
		hs.Market = probe("market", func() error {
			_, err := sc.Market.GetQuote(ctx, &pbm.QuoteRequest{Symbol: "SPY"})
			return err
		})
	}
	if sc.Strategy != nil {
		hs.Strategy = probe("strategy", func() error {
			_, err := sc.Strategy.GetBHState(ctx, &pbs.BHStateRequest{Symbol: "SPY", Timeframe: "1d"})
			return err
		})
	}
	if sc.Risk != nil {
		hs.Risk = probe("risk", func() error {
			_, err := sc.Risk.GetVaR(ctx, &pbr.VaRRequest{AccountId: "health-check", Confidence: 0.99, HorizonDays: 1})
			return err
		})
	}
	if sc.Portfolio != nil {
		hs.Portfolio = probe("portfolio", func() error {
			_, err := sc.Portfolio.GetPositions(ctx, &pbp.PositionsRequest{AccountId: "health-check"})
			return err
		})
	}
	return hs
}

// ─────────────────────────────────────────────────────────────────────────────
// Utility: backoff calculator (exported for use by callers)
// ─────────────────────────────────────────────────────────────────────────────

// ExponentialBackoff returns the delay for attempt n (0-indexed) with full jitter.
func ExponentialBackoff(base, maxBackoff time.Duration, multiplier float64, n int) time.Duration {
	exp := math.Pow(multiplier, float64(n))
	d := time.Duration(float64(base) * exp)
	if d > maxBackoff {
		d = maxBackoff
	}
	// Full jitter: random in [0, d].
	return time.Duration(rand.Int63n(int64(d) + 1))
}
