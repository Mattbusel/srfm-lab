FROM golang:1.22-alpine AS builder

WORKDIR /build
RUN apk add --no-cache git ca-certificates

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" \
    -o /build/monitor \
    ./cmd/monitor

# ---- Final stage ----
FROM alpine:3.19

RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app

COPY --from=builder /build/monitor /app/monitor

RUN mkdir -p /data

EXPOSE 8081

ENTRYPOINT ["/app/monitor"]
