FROM golang:1.22-alpine AS builder

WORKDIR /build

# Install build dependencies.
RUN apk add --no-cache git ca-certificates

# Copy go.mod and go.sum first to leverage build cache.
COPY go.mod go.sum ./
RUN go mod download

# Copy source.
COPY . .

# Build the gateway binary.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s -X main.version=$(date +%Y%m%d)" \
    -o /build/gateway \
    ./cmd/gateway

# ---- Final stage ----
FROM alpine:3.19

RUN apk add --no-cache ca-certificates tzdata wget

WORKDIR /app

# Copy the binary.
COPY --from=builder /build/gateway /app/gateway

# Create data directories.
RUN mkdir -p /data/parquet /data/cache

EXPOSE 8080

ENTRYPOINT ["/app/gateway"]
