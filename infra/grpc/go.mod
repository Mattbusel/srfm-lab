module github.com/srfm/infra/grpc

go 1.22

require (
	google.golang.org/grpc v1.64.0
	google.golang.org/protobuf v1.34.2
	github.com/prometheus/client_golang v1.19.1
	go.opentelemetry.io/otel v1.27.0
	go.opentelemetry.io/otel/trace v1.27.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc v1.27.0
	go.opentelemetry.io/otel/sdk v1.27.0
	go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.52.0
	github.com/golang-jwt/jwt/v5 v5.2.1
	github.com/sony/gobreaker v0.5.0
	golang.org/x/time v0.5.0
	github.com/parquet-go/parquet-go v0.23.0
	github.com/redis/go-redis/v9 v9.5.3
	go.uber.org/zap v1.27.0
)
