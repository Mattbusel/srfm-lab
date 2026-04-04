// metrics.go — Prometheus metrics gRPC interceptor.
package middleware

import (
	"context"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ─────────────────────────────────────────────────────────────────────────────
// Prometheus metrics registry
// ─────────────────────────────────────────────────────────────────────────────

var (
	grpcRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "grpc",
		Name:      "requests_total",
		Help:      "Total number of gRPC requests by method and status code.",
	}, []string{"method", "code"})

	grpcRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "srfm",
		Subsystem: "grpc",
		Name:      "request_duration_seconds",
		Help:      "gRPC request latency distribution.",
		Buckets:   []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
	}, []string{"method"})

	grpcRequestsInFlight = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "grpc",
		Name:      "requests_in_flight",
		Help:      "Number of gRPC requests currently being handled.",
	}, []string{"method"})

	grpcStreamMsgSent = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "grpc",
		Name:      "stream_messages_sent_total",
		Help:      "Total messages sent on server-side streams.",
	}, []string{"method"})

	grpcStreamMsgReceived = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "grpc",
		Name:      "stream_messages_received_total",
		Help:      "Total messages received on server-side streams.",
	}, []string{"method"})

	grpcStreamDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "srfm",
		Subsystem: "grpc",
		Name:      "stream_duration_seconds",
		Help:      "Duration of gRPC streaming RPCs.",
		Buckets:   []float64{1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600},
	}, []string{"method"})
)

// ─────────────────────────────────────────────────────────────────────────────
// Unary metrics interceptor
// ─────────────────────────────────────────────────────────────────────────────

// MetricsInterceptor returns a unary gRPC interceptor that records Prometheus metrics.
func MetricsInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		method := info.FullMethod
		start := time.Now()

		grpcRequestsInFlight.WithLabelValues(method).Inc()
		defer grpcRequestsInFlight.WithLabelValues(method).Dec()

		resp, err := handler(ctx, req)

		code := codes.OK
		if err != nil {
			if s, ok := status.FromError(err); ok {
				code = s.Code()
			} else {
				code = codes.Internal
			}
		}

		elapsed := time.Since(start).Seconds()
		grpcRequestDuration.WithLabelValues(method).Observe(elapsed)
		grpcRequestsTotal.WithLabelValues(method, strconv.Itoa(int(code))).Inc()

		return resp, err
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream metrics interceptor
// ─────────────────────────────────────────────────────────────────────────────

// MetricsStreamInterceptor returns a streaming gRPC interceptor that records metrics.
func MetricsStreamInterceptor() grpc.StreamServerInterceptor {
	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		method := info.FullMethod
		start := time.Now()

		wrapped := &metricsStream{
			ServerStream: ss,
			method:       method,
		}

		err := handler(srv, wrapped)

		code := codes.OK
		if err != nil {
			if s, ok := status.FromError(err); ok {
				code = s.Code()
			} else {
				code = codes.Internal
			}
		}

		elapsed := time.Since(start).Seconds()
		grpcStreamDuration.WithLabelValues(method).Observe(elapsed)
		grpcRequestsTotal.WithLabelValues(method, strconv.Itoa(int(code))).Inc()

		return err
	}
}

// metricsStream wraps grpc.ServerStream to count messages.
type metricsStream struct {
	grpc.ServerStream
	method string
}

func (s *metricsStream) SendMsg(m interface{}) error {
	err := s.ServerStream.SendMsg(m)
	if err == nil {
		grpcStreamMsgSent.WithLabelValues(s.method).Inc()
	}
	return err
}

func (s *metricsStream) RecvMsg(m interface{}) error {
	err := s.ServerStream.RecvMsg(m)
	if err == nil {
		grpcStreamMsgReceived.WithLabelValues(s.method).Inc()
	}
	return err
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom business metrics
// ─────────────────────────────────────────────────────────────────────────────

var (
	// Signal generation metrics.
	SignalComputeLatency = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "srfm",
		Subsystem: "strategy",
		Name:      "signal_compute_latency_us",
		Help:      "Time to compute a BH signal in microseconds.",
		Buckets:   prometheus.LinearBuckets(0, 500, 20), // 0–10,000 µs
	}, []string{"symbol", "timeframe", "strategy_id"})

	SignalsGeneratedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "strategy",
		Name:      "signals_generated_total",
		Help:      "Total signals generated by direction.",
	}, []string{"symbol", "direction", "strategy_id"})

	// Risk metrics.
	PreTradeChecksTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "risk",
		Name:      "pretrade_checks_total",
		Help:      "Total pre-trade risk checks by decision.",
	}, []string{"account_id", "decision"})

	RiskBreachesTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "risk",
		Name:      "breaches_total",
		Help:      "Total risk rule breaches by rule.",
	}, []string{"account_id", "rule_id", "severity"})

	// Portfolio metrics.
	PortfolioValue = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "portfolio",
		Name:      "equity_dollars",
		Help:      "Current portfolio equity in dollars.",
	}, []string{"account_id"})

	PortfolioPnL = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: "srfm",
		Subsystem: "portfolio",
		Name:      "pnl_dollars",
		Help:      "Current unrealized + realized PnL in dollars.",
	}, []string{"account_id"})

	ExecutionsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "portfolio",
		Name:      "executions_total",
		Help:      "Total trade executions recorded.",
	}, []string{"account_id", "side"})

	// Market data metrics.
	BarsFetchedTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "market",
		Name:      "bars_fetched_total",
		Help:      "Total bars fetched from cache.",
	}, []string{"symbol", "timeframe", "source"})

	QuoteRequestsTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "srfm",
		Subsystem: "market",
		Name:      "quote_requests_total",
		Help:      "Total quote requests.",
	}, []string{"symbol"})
)

// RecordSignalComputed records metrics for a completed signal computation.
func RecordSignalComputed(symbol, timeframe, strategyID, direction string, latencyUS int64) {
	SignalComputeLatency.WithLabelValues(symbol, timeframe, strategyID).Observe(float64(latencyUS))
	SignalsGeneratedTotal.WithLabelValues(symbol, direction, strategyID).Inc()
}

// RecordPreTrade records a pre-trade check outcome.
func RecordPreTrade(accountID, decision string) {
	PreTradeChecksTotal.WithLabelValues(accountID, decision).Inc()
}

// RecordRiskBreach records a risk rule breach event.
func RecordRiskBreach(accountID, ruleID, severity string) {
	RiskBreachesTotal.WithLabelValues(accountID, ruleID, severity).Inc()
}

// UpdatePortfolioMetrics updates portfolio gauge metrics.
func UpdatePortfolioMetrics(accountID string, equity, pnl float64) {
	PortfolioValue.WithLabelValues(accountID).Set(equity)
	PortfolioPnL.WithLabelValues(accountID).Set(pnl)
}
