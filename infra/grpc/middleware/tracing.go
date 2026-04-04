// tracing.go — OpenTelemetry distributed tracing gRPC interceptors.
package middleware

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
	"go.opentelemetry.io/otel/trace"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	grpccodes "google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const tracerName = "srfm.infra.grpc"

// ─────────────────────────────────────────────────────────────────────────────
// TracerProvider setup
// ─────────────────────────────────────────────────────────────────────────────

// TracingConfig configures the OpenTelemetry TracerProvider.
type TracingConfig struct {
	ServiceName    string
	ServiceVersion string
	OTLPEndpoint   string // e.g. "localhost:4317"
	SamplingRatio  float64
}

// InitTracerProvider initialises an OTLP gRPC exporter and returns a
// TracerProvider.  Call the returned shutdown func in your main's defer.
func InitTracerProvider(ctx context.Context, cfg TracingConfig, log *zap.Logger) (*sdktrace.TracerProvider, func(context.Context) error, error) {
	if cfg.ServiceName == "" {
		cfg.ServiceName = "srfm-grpc"
	}
	if cfg.ServiceVersion == "" {
		cfg.ServiceVersion = "1.0.0"
	}
	if cfg.SamplingRatio <= 0 {
		cfg.SamplingRatio = 1.0
	}

	// Exporter.
	var exporter sdktrace.SpanExporter
	if cfg.OTLPEndpoint != "" {
		exp, err := otlptracegrpc.New(ctx,
			otlptracegrpc.WithEndpoint(cfg.OTLPEndpoint),
			otlptracegrpc.WithInsecure(),
		)
		if err != nil {
			log.Warn("OTLP exporter init failed — using no-op", zap.Error(err))
		} else {
			exporter = exp
		}
	}

	// Resource.
	res, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName(cfg.ServiceName),
			semconv.ServiceVersion(cfg.ServiceVersion),
		),
	)
	if err != nil {
		res = resource.Default()
	}

	// Build provider.
	opts := []sdktrace.TracerProviderOption{
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.TraceIDRatioBased(cfg.SamplingRatio)),
	}
	if exporter != nil {
		opts = append(opts, sdktrace.WithBatcher(exporter))
	}

	tp := sdktrace.NewTracerProvider(opts...)
	otel.SetTracerProvider(tp)

	shutdown := func(ctx context.Context) error {
		return tp.Shutdown(ctx)
	}
	return tp, shutdown, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Propagation helpers
// ─────────────────────────────────────────────────────────────────────────────

// mdCarrier implements propagation.TextMapCarrier for gRPC metadata.
type mdCarrier metadata.MD

func (c mdCarrier) Get(key string) string {
	vals := metadata.MD(c).Get(key)
	if len(vals) == 0 {
		return ""
	}
	return vals[0]
}

func (c mdCarrier) Set(key, val string) {
	metadata.MD(c).Set(key, val)
}

func (c mdCarrier) Keys() []string {
	keys := make([]string, 0, len(c))
	for k := range c {
		keys = append(keys, k)
	}
	return keys
}

// ─────────────────────────────────────────────────────────────────────────────
// Unary tracing interceptor
// ─────────────────────────────────────────────────────────────────────────────

// TracingInterceptor returns a unary gRPC interceptor that creates and propagates
// OpenTelemetry spans.
func TracingInterceptor() grpc.UnaryServerInterceptor {
	tracer := otel.Tracer(tracerName)

	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		// Extract trace context from incoming metadata.
		md, ok := metadata.FromIncomingContext(ctx)
		if ok {
			ctx = otel.GetTextMapPropagator().Extract(ctx, mdCarrier(md))
		}

		spanName := info.FullMethod
		ctx, span := tracer.Start(ctx, spanName,
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(
				semconv.RPCSystem("grpc"),
				semconv.RPCMethod(methodName(info.FullMethod)),
				semconv.RPCService(serviceName(info.FullMethod)),
			),
		)
		defer span.End()

		// Add auth metadata to span.
		if claims, ok := ClaimsFromContext(ctx); ok {
			span.SetAttributes(attribute.String("auth.subject", claims.Subject))
			span.SetAttributes(attribute.String("auth.account_id", claims.AccountID))
		}

		resp, err := handler(ctx, req)

		// Record error status.
		if err != nil {
			if s, ok := status.FromError(err); ok {
				span.SetStatus(otlpStatusFromGRPC(s.Code()), s.Message())
				span.SetAttributes(attribute.Int("grpc.status_code", int(s.Code())))
			} else {
				span.SetStatus(codes.Error, err.Error())
			}
		} else {
			span.SetStatus(codes.Ok, "")
		}

		return resp, err
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream tracing interceptor
// ─────────────────────────────────────────────────────────────────────────────

// TracingStreamInterceptor returns a streaming gRPC interceptor that creates spans.
func TracingStreamInterceptor() grpc.StreamServerInterceptor {
	tracer := otel.Tracer(tracerName)

	return func(srv interface{}, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		ctx := ss.Context()
		md, ok := metadata.FromIncomingContext(ctx)
		if ok {
			ctx = otel.GetTextMapPropagator().Extract(ctx, mdCarrier(md))
		}

		ctx, span := tracer.Start(ctx, info.FullMethod,
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(
				semconv.RPCSystem("grpc"),
				semconv.RPCMethod(methodName(info.FullMethod)),
				semconv.RPCService(serviceName(info.FullMethod)),
				attribute.Bool("grpc.is_client_stream", info.IsClientStream),
				attribute.Bool("grpc.is_server_stream", info.IsServerStream),
			),
		)
		defer span.End()

		tracingStream := &tracingServerStream{
			ServerStream: &wrappedStream{ServerStream: ss, ctx: ctx},
			span:         span,
			method:       info.FullMethod,
		}

		err := handler(srv, tracingStream)
		if err != nil {
			if s, ok := status.FromError(err); ok {
				span.SetStatus(otlpStatusFromGRPC(s.Code()), s.Message())
			} else {
				span.SetStatus(codes.Error, err.Error())
			}
		}
		return err
	}
}

// tracingServerStream wraps grpc.ServerStream to add span events per message.
type tracingServerStream struct {
	grpc.ServerStream
	span   trace.Span
	method string
	sentCount     int
	receivedCount int
}

func (s *tracingServerStream) SendMsg(m interface{}) error {
	err := s.ServerStream.SendMsg(m)
	s.sentCount++
	s.span.AddEvent("message.sent", trace.WithAttributes(
		attribute.Int("message.count", s.sentCount),
	))
	return err
}

func (s *tracingServerStream) RecvMsg(m interface{}) error {
	err := s.ServerStream.RecvMsg(m)
	s.receivedCount++
	s.span.AddEvent("message.received", trace.WithAttributes(
		attribute.Int("message.count", s.receivedCount),
	))
	return err
}

// ─────────────────────────────────────────────────────────────────────────────
// Client-side tracing interceptor
// ─────────────────────────────────────────────────────────────────────────────

// ClientTracingInterceptor adds trace propagation to outgoing gRPC calls.
func ClientTracingInterceptor() grpc.UnaryClientInterceptor {
	tracer := otel.Tracer(tracerName + ".client")

	return func(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		ctx, span := tracer.Start(ctx, method,
			trace.WithSpanKind(trace.SpanKindClient),
			trace.WithAttributes(
				semconv.RPCSystem("grpc"),
				semconv.RPCMethod(methodName(method)),
				semconv.RPCService(serviceName(method)),
			),
		)
		defer span.End()

		// Inject trace context into outgoing metadata.
		md, ok := metadata.FromOutgoingContext(ctx)
		if !ok {
			md = metadata.MD{}
		}
		otel.GetTextMapPropagator().Inject(ctx, mdCarrier(md))
		ctx = metadata.NewOutgoingContext(ctx, md)

		err := invoker(ctx, method, req, reply, cc, opts...)
		if err != nil {
			if s, ok := status.FromError(err); ok {
				span.SetStatus(otlpStatusFromGRPC(s.Code()), s.Message())
			} else {
				span.SetStatus(codes.Error, err.Error())
			}
		}
		return err
	}
}

// ClientStreamTracingInterceptor adds trace propagation to outgoing streaming calls.
func ClientStreamTracingInterceptor() grpc.StreamClientInterceptor {
	tracer := otel.Tracer(tracerName + ".client")

	return func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
		ctx, span := tracer.Start(ctx, method,
			trace.WithSpanKind(trace.SpanKindClient),
			trace.WithAttributes(
				semconv.RPCSystem("grpc"),
				semconv.RPCMethod(methodName(method)),
				semconv.RPCService(serviceName(method)),
			),
		)

		md, ok := metadata.FromOutgoingContext(ctx)
		if !ok {
			md = metadata.MD{}
		}
		otel.GetTextMapPropagator().Inject(ctx, mdCarrier(md))
		ctx = metadata.NewOutgoingContext(ctx, md)

		cs, err := streamer(ctx, desc, cc, method, opts...)
		if err != nil {
			span.SetStatus(codes.Error, err.Error())
			span.End()
			return nil, err
		}

		return &tracingClientStream{ClientStream: cs, span: span}, nil
	}
}

type tracingClientStream struct {
	grpc.ClientStream
	span trace.Span
}

func (s *tracingClientStream) CloseSend() error {
	err := s.ClientStream.CloseSend()
	s.span.End()
	return err
}

func (s *tracingClientStream) RecvMsg(m interface{}) error {
	err := s.ClientStream.RecvMsg(m)
	if err != nil {
		s.span.End()
	}
	return err
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func otlpStatusFromGRPC(code grpccodes.Code) codes.Code {
	if code == grpccodes.OK {
		return codes.Ok
	}
	return codes.Error
}

// methodName extracts the method name from a full gRPC method path.
// e.g. "/srfm.market.MarketDataService/GetBars" → "GetBars"
func methodName(fullMethod string) string {
	if idx := len(fullMethod) - 1; idx >= 0 {
		for i := idx; i >= 0; i-- {
			if fullMethod[i] == '/' {
				return fullMethod[i+1:]
			}
		}
	}
	return fullMethod
}

// serviceName extracts the service name from a full gRPC method path.
// e.g. "/srfm.market.MarketDataService/GetBars" → "srfm.market.MarketDataService"
func serviceName(fullMethod string) string {
	if len(fullMethod) == 0 {
		return ""
	}
	s := fullMethod
	if s[0] == '/' {
		s = s[1:]
	}
	for i := 0; i < len(s); i++ {
		if s[i] == '/' {
			return s[:i]
		}
	}
	return s
}

// SpanFromContext returns the active span in the context (or a no-op span).
func SpanFromContext(ctx context.Context) trace.Span {
	return trace.SpanFromContext(ctx)
}

// AddSpanEvent is a convenience wrapper to add an event to the active span.
func AddSpanEvent(ctx context.Context, name string, attrs ...attribute.KeyValue) {
	trace.SpanFromContext(ctx).AddEvent(name, trace.WithAttributes(attrs...))
}

// SetSpanError records an error on the active span.
func SetSpanError(ctx context.Context, err error) {
	span := trace.SpanFromContext(ctx)
	span.SetStatus(codes.Error, fmt.Sprintf("%v", err))
}
