// server.go — gRPC server assembly: builds and registers all four services,
// configures TLS, middleware chain, graceful shutdown, and health checking.
package server

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/reflection"

	pbm  "github.com/srfm/infra/grpc/proto/market"
	pbp  "github.com/srfm/infra/grpc/proto/portfolio"
	pbr  "github.com/srfm/infra/grpc/proto/risk"
	pbs  "github.com/srfm/infra/grpc/proto/strategy"
	"github.com/srfm/infra/grpc/middleware"
)

// ─────────────────────────────────────────────────────────────────────────────
// Server configuration
// ─────────────────────────────────────────────────────────────────────────────

// Config bundles all configuration for the gRPC server assembly.
type Config struct {
	// ListenAddr is the TCP address to listen on, e.g. ":50051".
	ListenAddr string

	// MetricsAddr is the HTTP address for Prometheus /metrics endpoint.
	// Empty string disables the metrics server.
	MetricsAddr string

	// TLS configuration.
	TLSCertFile string
	TLSKeyFile  string
	TLSCACert   string // for mTLS client verification

	// Auth configuration.
	Auth middleware.AuthConfig

	// Rate limiter.
	RateLimit middleware.RateLimiterConfig

	// Tracing.
	Tracing middleware.TracingConfig

	// Service configurations.
	Market    MarketServerConfig
	Strategy  StrategyServerConfig
	Risk      RiskServerConfig

	// gRPC server tunables.
	MaxRecvMsgSize     int
	MaxSendMsgSize     int
	MaxConcurrentStreams uint32
	KeepaliveTime      time.Duration
	KeepaliveTimeout   time.Duration
	KeepaliveMinTime   time.Duration

	// GracefulShutdownTimeout is how long to wait for in-flight RPCs on shutdown.
	GracefulShutdownTimeout time.Duration

	// EnableReflection enables gRPC server reflection (useful for grpcurl).
	EnableReflection bool
}

// DefaultConfig returns a config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		ListenAddr:              ":50051",
		MetricsAddr:             ":9090",
		Market:                  DefaultMarketServerConfig(),
		Strategy:                DefaultStrategyServerConfig(),
		Risk:                    DefaultRiskServerConfig(),
		MaxRecvMsgSize:          32 * 1024 * 1024, // 32 MiB
		MaxSendMsgSize:          32 * 1024 * 1024,
		MaxConcurrentStreams:    1000,
		KeepaliveTime:           30 * time.Second,
		KeepaliveTimeout:        10 * time.Second,
		KeepaliveMinTime:        5 * time.Second,
		GracefulShutdownTimeout: 30 * time.Second,
		EnableReflection:        true,
		Auth: middleware.AuthConfig{
			AllowAnonymous: true,
		},
		RateLimit: middleware.RateLimiterConfig{
			GlobalRPS:       10_000,
			PerClientRPS:    1_000,
			BurstMultiplier: 3,
		},
		Tracing: middleware.TracingConfig{
			ServiceName:   "srfm-grpc",
			SamplingRatio: 1.0,
		},
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Assembly — wires together all services
// ─────────────────────────────────────────────────────────────────────────────

// Assembly holds all the constructed services and the gRPC server.
type Assembly struct {
	cfg        Config
	log        *zap.Logger
	grpcServer *grpc.Server
	httpServer *http.Server

	// Services.
	Market    *MarketServer
	Strategy  *StrategyServer
	Risk      *RiskServer
	Portfolio *PortfolioServer
}

// NewAssembly constructs all services and wires them together.
func NewAssembly(cfg Config, log *zap.Logger) (*Assembly, error) {
	// Build services in dependency order.
	marketSrv := NewMarketServer(cfg.Market, log)
	portfolioSrv := NewPortfolioServer(marketSrv, log)
	strategySrv := NewStrategyServer(cfg.Strategy, marketSrv, log)
	riskSrv := NewRiskServer(cfg.Risk, marketSrv, portfolioSrv, log)

	// Build gRPC server options.
	grpcOpts, err := buildServerOptions(cfg, log)
	if err != nil {
		return nil, fmt.Errorf("build gRPC options: %w", err)
	}

	grpcSrv := grpc.NewServer(grpcOpts...)

	// Register service implementations.
	pbm.RegisterMarketDataServiceServer(grpcSrv, marketSrv)
	pbs.RegisterStrategyServiceServer(grpcSrv, strategySrv)
	pbr.RegisterRiskServiceServer(grpcSrv, riskSrv)
	pbp.RegisterPortfolioServiceServer(grpcSrv, portfolioSrv)

	// Health checking.
	healthSrv := health.NewServer()
	healthpb.RegisterHealthServer(grpcSrv, healthSrv)
	healthSrv.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	healthSrv.SetServingStatus("srfm.market.MarketDataService", healthpb.HealthCheckResponse_SERVING)
	healthSrv.SetServingStatus("srfm.strategy.StrategyService", healthpb.HealthCheckResponse_SERVING)
	healthSrv.SetServingStatus("srfm.risk.RiskService", healthpb.HealthCheckResponse_SERVING)
	healthSrv.SetServingStatus("srfm.portfolio.PortfolioService", healthpb.HealthCheckResponse_SERVING)

	// Server reflection (gRPC introspection).
	if cfg.EnableReflection {
		reflection.Register(grpcSrv)
	}

	// Metrics HTTP server.
	var httpSrv *http.Server
	if cfg.MetricsAddr != "" {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.Handler())
		mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok"))
		})
		httpSrv = &http.Server{
			Addr:         cfg.MetricsAddr,
			Handler:      mux,
			ReadTimeout:  10 * time.Second,
			WriteTimeout: 10 * time.Second,
		}
	}

	return &Assembly{
		cfg:        cfg,
		log:        log,
		grpcServer: grpcSrv,
		httpServer: httpSrv,
		Market:     marketSrv,
		Strategy:   strategySrv,
		Risk:       riskSrv,
		Portfolio:  portfolioSrv,
	}, nil
}

// Run starts the gRPC server and metrics HTTP server, blocks until a shutdown
// signal is received, then gracefully drains in-flight RPCs.
func (a *Assembly) Run() error {
	lis, err := net.Listen("tcp", a.cfg.ListenAddr)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", a.cfg.ListenAddr, err)
	}

	// Metrics server.
	if a.httpServer != nil {
		go func() {
			a.log.Info("metrics server starting", zap.String("addr", a.cfg.MetricsAddr))
			if err := a.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				a.log.Error("metrics server error", zap.Error(err))
			}
		}()
	}

	// gRPC server.
	go func() {
		a.log.Info("gRPC server starting", zap.String("addr", a.cfg.ListenAddr))
		if err := a.grpcServer.Serve(lis); err != nil {
			a.log.Error("gRPC server error", zap.Error(err))
		}
	}()

	// Wait for shutdown signal.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	sig := <-quit
	a.log.Info("shutdown signal received", zap.String("signal", sig.String()))

	return a.Shutdown()
}

// Shutdown performs a graceful shutdown.
func (a *Assembly) Shutdown() error {
	ctx, cancel := context.WithTimeout(context.Background(), a.cfg.GracefulShutdownTimeout)
	defer cancel()

	// Gracefully stop gRPC (waits for in-flight RPCs to complete).
	stopped := make(chan struct{})
	go func() {
		a.grpcServer.GracefulStop()
		close(stopped)
	}()

	select {
	case <-stopped:
		a.log.Info("gRPC server stopped gracefully")
	case <-ctx.Done():
		a.log.Warn("graceful shutdown timeout, forcing stop")
		a.grpcServer.Stop()
	}

	// Stop metrics server.
	if a.httpServer != nil {
		shutCtx, shutCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutCancel()
		_ = a.httpServer.Shutdown(shutCtx)
	}

	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// gRPC server option builder
// ─────────────────────────────────────────────────────────────────────────────

func buildServerOptions(cfg Config, log *zap.Logger) ([]grpc.ServerOption, error) {
	var opts []grpc.ServerOption

	// TLS.
	if cfg.TLSCertFile != "" && cfg.TLSKeyFile != "" {
		creds, err := buildServerTLS(cfg)
		if err != nil {
			return nil, err
		}
		opts = append(opts, grpc.Creds(creds))
	}

	// Message size limits.
	opts = append(opts,
		grpc.MaxRecvMsgSize(cfg.MaxRecvMsgSize),
		grpc.MaxSendMsgSize(cfg.MaxSendMsgSize),
	)

	// Connection parameters.
	opts = append(opts, grpc.KeepaliveParams(keepalive.ServerParameters{
		MaxConnectionIdle:     15 * time.Minute,
		MaxConnectionAge:      30 * time.Minute,
		MaxConnectionAgeGrace: 5 * time.Second,
		Time:                  cfg.KeepaliveTime,
		Timeout:               cfg.KeepaliveTimeout,
	}))
	opts = append(opts, grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{
		MinTime:             cfg.KeepaliveMinTime,
		PermitWithoutStream: true,
	}))
	if cfg.MaxConcurrentStreams > 0 {
		opts = append(opts, grpc.MaxConcurrentStreams(cfg.MaxConcurrentStreams))
	}

	// Build unary interceptor chain.
	unaryChain := middleware.ChainUnary(
		middleware.TracingInterceptor(),
		middleware.MetricsInterceptor(),
		middleware.AuthInterceptor(cfg.Auth, log),
		middleware.RateLimiterInterceptor(cfg.RateLimit),
	)

	// Build stream interceptor chain.
	streamChain := middleware.ChainStream(
		middleware.TracingStreamInterceptor(),
		middleware.MetricsStreamInterceptor(),
		middleware.AuthStreamInterceptor(cfg.Auth, log),
		middleware.RateLimiterStreamInterceptor(cfg.RateLimit),
	)

	opts = append(opts,
		grpc.UnaryInterceptor(unaryChain),
		grpc.StreamInterceptor(streamChain),
	)

	// OTel stats handler (spans for all calls).
	opts = append(opts, grpc.StatsHandler(otelgrpc.NewServerHandler()))

	return opts, nil
}

// buildServerTLS loads TLS credentials for the gRPC server.
func buildServerTLS(cfg Config) (credentials.TransportCredentials, error) {
	cert, err := tls.LoadX509KeyPair(cfg.TLSCertFile, cfg.TLSKeyFile)
	if err != nil {
		return nil, fmt.Errorf("load server cert/key: %w", err)
	}

	tlsCfg := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS13,
	}

	// mTLS: require and verify client certificates.
	if cfg.TLSCACert != "" {
		caCert, err := os.ReadFile(cfg.TLSCACert)
		if err != nil {
			return nil, fmt.Errorf("read CA cert: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("parse CA cert")
		}
		tlsCfg.ClientCAs = pool
		tlsCfg.ClientAuth = tls.RequireAndVerifyClientCert
	}

	return credentials.NewTLS(tlsCfg), nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone main-like entrypoint (for embedding in cmd/)
// ─────────────────────────────────────────────────────────────────────────────

// RunWithDefaults starts the gRPC server with default configuration,
// overriding with any non-zero values from cfg.
func RunWithDefaults(overrides Config) error {
	log, err := zap.NewProduction()
	if err != nil {
		return fmt.Errorf("build logger: %w", err)
	}
	defer log.Sync() //nolint:errcheck

	cfg := DefaultConfig()
	// Apply overrides.
	if overrides.ListenAddr != "" {
		cfg.ListenAddr = overrides.ListenAddr
	}
	if overrides.MetricsAddr != "" {
		cfg.MetricsAddr = overrides.MetricsAddr
	}
	if overrides.TLSCertFile != "" {
		cfg.TLSCertFile = overrides.TLSCertFile
		cfg.TLSKeyFile = overrides.TLSKeyFile
	}

	// Initialise OpenTelemetry tracing.
	ctx := context.Background()
	_, shutdownTP, err := middleware.InitTracerProvider(ctx, cfg.Tracing, log)
	if err == nil {
		defer shutdownTP(ctx)
	}

	asm, err := NewAssembly(cfg, log)
	if err != nil {
		return fmt.Errorf("build assembly: %w", err)
	}

	return asm.Run()
}
