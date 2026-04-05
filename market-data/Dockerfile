FROM golang:1.21-alpine AS builder

RUN apk add --no-cache gcc musl-dev sqlite-dev

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=1 GOOS=linux go build -o market-data .

FROM alpine:3.19
RUN apk add --no-cache sqlite-libs ca-certificates

WORKDIR /app
COPY --from=builder /app/market-data .

ENV ALPACA_API_KEY=""
ENV ALPACA_API_SECRET=""

EXPOSE 8780
ENTRYPOINT ["./market-data"]
